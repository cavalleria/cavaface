import os
import sys
import time
import random
import numpy as np
import scipy
import pickle
import builtins
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from config import configurations
from backbone.resnet import *
from backbone.resnet_irse import *
from backbone.mobilefacenet import *
from backbone.ghostnet import *
from head.metrics import *
from loss.loss import *
from util.utils import *
from dataset.datasets import FaceDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from util.flops_counter import *
from optimizer.lr_scheduler import *
from optimizer.optimizer import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)

def main():
    cfg = configurations[1]
    SEED = cfg['SEED'] # random seed for reproduce results
    set_seed(int(SEED))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    ngpus_per_node = len(cfg['GPU'])
    world_size = cfg['WORLD_SIZE']
    cfg['WORLD_SIZE'] = ngpus_per_node * world_size
    VAL_DATA_ROOT = cfg['VAL_DATA_ROOT']
    if not isinstance(cfg['VAL_SET'], list):
        cfg['VAL_SET'] = [cfg['VAL_SET']]
    valdata = get_val_data(VAL_DATA_ROOT, cfg['VAL_SET'])

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, valdata))
    
def main_worker(gpu, ngpus_per_node, cfg, valdata):
    cfg['GPU'] = gpu
    if gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    cfg['RANK'] = cfg['RANK'] * ngpus_per_node + gpu
    dist.init_process_group(backend=cfg['DIST_BACKEND'], init_method = cfg["DIST_URL"], world_size=cfg['WORLD_SIZE'], rank=cfg['RANK'])
    
    # Data loading code
    batch_size = int(cfg['BATCH_SIZE'])
    per_batch_size = int(batch_size / ngpus_per_node)
    #workers = int((cfg['NUM_WORKERS'] + ngpus_per_node - 1) / ngpus_per_node) # dataload threads
    workers = int(cfg['NUM_WORKERS'])
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    VAL_DATA_ROOT = cfg['VAL_DATA_ROOT']
    RECORD_DIR = cfg['RECORD_DIR']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    DROP_LAST = cfg['DROP_LAST']
    LR_SCHEDULER = cfg['LR_SCHEDULER']
    LR_STEP_SIZE = cfg['LR_STEP_SIZE']
    LR_DECAY_EPOCH = cfg['LR_DECAY_EPOCH']
    LR_DECAT_GAMMA = cfg['LR_DECAT_GAMMA']
    LR_END = cfg['LR_END']
    WARMUP_EPOCH = cfg['WARMUP_EPOCH']
    WARMUP_LR = cfg['WARMUP_LR']
    NUM_EPOCH = cfg['NUM_EPOCH']
    USE_APEX = cfg['USE_APEX']
    EVAL_FREQ = cfg['EVAL_FREQ']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)
    train_transform = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize(mean = RGB_MEAN,
                            std = RGB_STD),])
    dataset_train = FaceDataset(DATA_ROOT, RECORD_DIR, train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=per_batch_size, shuffle = (train_sampler is None), num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=DROP_LAST)
    SAMPLE_NUMS = dataset_train.get_sample_num_of_each_class()
    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))
     
    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'MobileFaceNet': MobileFaceNet,
                     'ResNet_50': ResNet_50, 
                     'ResNet_101': ResNet_101, 
                     'ResNet_152': ResNet_152,
                     'IR_50': IR_50, 
                     'IR_100': IR_100,
                     'IR_101': IR_101, 
                     'IR_152': IR_152,
                     'IR_185': IR_185,
                     'IR_200': IR_200,
                     'IR_SE_50': IR_SE_50, 
                     'IR_SE_100': IR_SE_100,
                     'IR_SE_101': IR_SE_101, 
                     'IR_SE_152': IR_SE_152,
                     'IR_SE_185': IR_SE_185,
                     'IR_SE_200': IR_SE_200,
                     'GhostNet': ghost_net}
    BACKBONE_NAME = cfg['BACKBONE_NAME']
    INPUT_SIZE = cfg['INPUT_SIZE']
    assert INPUT_SIZE == [112, 112]
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    backbone = BACKBONE_DICT[BACKBONE_NAME](INPUT_SIZE, emb_size=EMBEDDING_SIZE, neck_type=cfg["NECK"])
    print("=" * 60)
    print(backbone)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)
    HEAD_DICT = {'Softmax': Softmax,
                 'ArcFace': ArcFace,
                 'CosFace': CosFace,
                 'SphereFace': SphereFace,
                 'Am_softmax': Am_softmax,
                 'CurricularFace': CurricularFace,
                 'ArcNegFace': ArcNegFace,
                 'SVX': SVXSoftmax,
                 'AirFace': AirFace,
                 'QAMFace': QAMFace}
    HEAD_NAME = cfg['HEAD_NAME']
    head = HEAD_DICT[HEAD_NAME](in_features = EMBEDDING_SIZE, out_features = NUM_CLASS)
    print("Params: ", count_model_params(backbone))
    print("Flops:", count_model_flops(backbone))
    print("=" * 60)
    print(head)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    
   #--------------------optimizer-----------------------------
    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(backbone) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    
    LR = cfg['LR'] # initial LR
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    optimizer = eval('optm_'+cfg['OPTIMIZER'])(cfg, 
                            [{'params': backbone_paras_wo_bn + list(head.parameters()), 'weight_decay': WEIGHT_DECAY}, 
                            {'params': backbone_paras_only_bn}
                            ])
    
    scheduler = None
    if cfg['OPTIMIZER'] in ['SGD']: # non-adaptive, set learning schedule
        if LR_SCHEDULER == 'step':
            scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAT_GAMMA)
        elif LR_SCHEDULER == 'multi_step':
            scheduler = MultiStepLR(optimizer, milestones=LR_DECAY_EPOCH, gamma=LR_DECAT_GAMMA)
        elif LR_SCHEDULER == 'cosine':
            scheduler = CosineWarmupLR(optimizer, batches=len(train_loader), epochs=NUM_EPOCH, base_lr=LR, target_lr=LR_END, warmup_epochs=WARMUP_EPOCH, warmup_lr=WARMUP_LR)
        
    print("=" * 60)
    print(optimizer)
    print("Optimizer Generated")
    print("=" * 60)
  
    # loss
    LOSS_NAME = cfg['LOSS_NAME']
    LOSS_DICT = {'Softmax'      : nn.CrossEntropyLoss(),
                 'LabelSmooth'  : LabelSmoothCrossEntropyLoss(classes=NUM_CLASS,smoothing=0.1),
                 'Focal'        : FocalLoss(),
                 'HM'           : HardMining()}
    loss = LOSS_DICT[LOSS_NAME].cuda(gpu)
    print("=" * 60)
    print(loss)
    print("{} Loss Generated".format(loss))
    print("=" * 60)
    
    torch.cuda.set_device(cfg['GPU'])
    backbone.cuda(cfg['GPU'])
    head.cuda(cfg['GPU'])

    #optionally resume from a checkpoint 
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    IS_RESUME = cfg['IS_RESUME']
    if IS_RESUME:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            loc = 'cuda:{}'.format(cfg['GPU'])
            backbone.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, map_location=loc))
            if os.path.isfile(HEAD_RESUME_ROOT):
                print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
                checkpoint = torch.load(HEAD_RESUME_ROOT, map_location=loc)
                cfg['START_EPOCH'] = checkpoint['EPOCH']
                head.load_state_dict(checkpoint['HEAD'])
                optimizer.load_state_dict(checkpoint['OPTIMIZER'])
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    
    if USE_APEX:
        [backbone, head], optimizer = amp.initialize([backbone, head], optimizer, opt_level='O2')
        backbone = DDP(backbone)
        head = DDP(head)
    else:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[cfg['GPU']])
        head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[cfg['GPU']])

     # checkpoint and tensorboard dir
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    
    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)

    writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results

    def evaluate():
        print("=" * 60)
        print("Perform Evaluation on %s, and Save Checkpoints..."%(','.join([v[2] for v in valdata])))
        sys.stdout.flush()
        acc_str = ""
        for v in valdata:
            accuracy, best_threshold, roc_curve = perform_val(EMBEDDING_SIZE, batch_size, backbone, v[0], v[1])
            buffer_val(writer, "%s"%(v[2]), accuracy, best_threshold, roc_curve, epoch + 1)
            print("%s eval done"%(v[2]))
            sys.stdout.flush()
            acc_str = acc_str + "%s Acc: %.4f, "%(v[2], accuracy*100)
        print("Epoch {}/{}, Evaluation: {}".format(epoch + 1, NUM_EPOCH, acc_str))
        print("=" * 60)
        sys.stdout.flush()

    # train
    global_step = 0
    for epoch in range(cfg['START_EPOCH'], cfg['NUM_EPOCH']):
        train_sampler.set_epoch(epoch)
        if LR_SCHEDULER is not None and LR_SCHEDULER != 'cosine':
            scheduler.step()
        #train for one epoch
        #train(train_loader, backbone, head, loss, optimizer, epoch, cfg, writer)
        DISP_FREQ = -1  # 100 batch
        EVAL_FREQ = cfg["EVAL_FREQ"]

        batch = 0  # batch index
        backbone.train()  # set to training mode
        head.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with tqdm(total=len(train_loader), desc=cfg['BACKBONE_NAME'], unit=' samples', unit_scale=batch_size, 
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {rate_fmt} | {postfix}', dynamic_ncols=True) as t:
            for inputs, labels in iter(train_loader):
                if LR_SCHEDULER is not None and LR_SCHEDULER == 'cosine':
                    scheduler.step()
                # compute output
                start_time=time.time()
                inputs = inputs.cuda(cfg['GPU'], non_blocking=True)
                labels = labels.cuda(cfg['GPU'], non_blocking=True)
                features, conv_features = backbone(inputs)

                outputs = head(features, labels)
                lossx = loss(outputs, labels)
                end_time = time.time()
                duration = end_time - start_time
                if DISP_FREQ > 0 and ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("batch inference time", duration)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                if USE_APEX:
                    with amp.scale_loss(lossx, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    lossx.backward()
                optimizer.step()
                
                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
                losses.update(lossx.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))
                # tqdm update
                t.set_postfix_str('batch {}/{} | loss={:.3f} top1={:.3f} top5={:.3f}'.format(
                                    batch, len(train_loader), lossx.data.item(), prec1.data.item(),prec5.data.item()))

                # dispaly training loss & acc every DISP_FREQ
                if DISP_FREQ > 0 and ((batch + 1) % DISP_FREQ == 0) or batch == 0:
                    print("=" * 60)
                    print('Epoch {}/{} Batch {}/{}\t'
                                    'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                    'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                    'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                        epoch + 1, cfg['NUM_EPOCH'], batch + 1, len(train_loader), loss = losses, top1 = top1, top5 = top5))
                    print("=" * 60)

                # perform validation & save checkpoints per epoch
                # validation statistics per epoch (buffer for visualization)
                if (global_step + 1) % EVAL_FREQ == 0:
                    #lr = scheduler.get_last_lr()
                    lr = optimizer.param_groups[0]['lr']
                    print("Current lr", lr)
                    evaluate()

                    print("=" * 60)
                    print("Save Checkpoint...")
                    if cfg['RANK'] % ngpus_per_node == 0:
                        torch.save(backbone.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, get_time())))
                        save_dict = {'EPOCH': epoch+1,
                                    'HEAD': head.module.state_dict(),
                                    'OPTIMIZER': optimizer.state_dict()}
                        torch.save(save_dict, os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, get_time())))
                sys.stdout.flush()
                batch += 1 # batch index
                global_step += 1
                t.update()
    
            epoch_loss = losses.avg
            epoch_acc = top1.avg
            print("=" * 60)
            print('Epoch: {}/{}\t''Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch + 1, cfg['NUM_EPOCH'], loss = losses, top1 = top1, top5 = top5))
            sys.stdout.flush()
            print("=" * 60)
            if cfg['RANK'] % ngpus_per_node == 0:
                writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
                writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
                writer.add_scalar("Top1", top1.avg, epoch+1)
                writer.add_scalar("Top5", top5.avg, epoch+1)
            
    
if __name__ == '__main__':
    main()
