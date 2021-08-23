import os
import sys
import copy
import apex
import torch
import builtins
from apex import amp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from config import configurations
from backbone import *
from head import *
from loss.loss import *
from dataset import *
from util.utils import *
from util.verification import *
from util.flops_counter import *
from optimizer.lr_scheduler import *
from optimizer.optimizer import *


def main():
    cfg = configurations[1]
    ngpus_per_node = len(cfg["GPU"])
    world_size = cfg["WORLD_SIZE"]
    cfg["WORLD_SIZE"] = ngpus_per_node * world_size

    # load val data
    val_dataset = get_val_dataset(cfg["DATA_ROOT"], cfg["VAL_SET"])
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))


def main_worker(gpu, ngpus_per_node, cfg):
    cfg["GPU"] = gpu
    SEED = cfg["SEED"]
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass
    cfg["RANK"] = cfg["RANK"] * ngpus_per_node + gpu
    dist.init_process_group(
        backend=cfg["DIST_BACKEND"],
        init_method=cfg["DIST_URL"],
        world_size=cfg["WORLD_SIZE"],
        rank=cfg["RANK"],
    )

    MODEL_ROOT = cfg["MODEL_ROOT"]
    os.makedirs(MODEL_ROOT, exist_ok=True)

    batch_size = int(cfg["BATCH_SIZE"])
    per_batch_size = int(batch_size / ngpus_per_node)
    workers = int(cfg["NUM_WORKERS"])
    DATA_ROOT = cfg["DATA_ROOT"]
    SYNC_DATA = cfg["SYNC_DATA"]
    SYNC_DATA_NUMCLASS = cfg["SYNC_DATA_NUMCLASS"]
    RGB_MEAN = cfg["RGB_MEAN"]
    RGB_STD = cfg["RGB_STD"]
    DROP_LAST = cfg["DROP_LAST"]
    OPTIMIZER = cfg["OPTIMIZER"]
    LR_SCHEDULER = cfg["LR_SCHEDULER"]
    LR_STEP_SIZE = cfg["LR_STEP_SIZE"]
    LR_DECAY_EPOCH = cfg["LR_DECAY_EPOCH"]
    LR_DECAT_GAMMA = cfg["LR_DECAT_GAMMA"]
    LR_END = cfg["LR_END"]
    WARMUP_EPOCH = cfg["WARMUP_EPOCH"]
    WARMUP_LR = cfg["WARMUP_LR"]
    NUM_EPOCH = cfg["NUM_EPOCH"]
    USE_AMP = cfg["USE_AMP"]
    EVAL_FREQ = cfg["EVAL_FREQ"]
    SYNC_BN = cfg["SYNC_BN"]
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)
    transform_list = [
        transforms.RandomHorizontalFlip(),
    ]
    if cfg["COLORJITTER"]:
        transform_list.append(
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        )
    if cfg["CUTOUT"]:
        transform_list.append(Cutout())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=RGB_MEAN, std=RGB_STD))
    if cfg["RANDOM_ERASING"]:
        transform_list.append(transforms.RandomErasing())
    train_transform = transforms.Compose(transform_list)
    if cfg["RANDAUGMENT"]:
        train_transform.transforms.insert(
            0, RandAugment(n=cfg["RANDAUGMENT_N"], m=cfg["RANDAUGMENT_M"])
        )
    print("=" * 60)
    print(train_transform)
    print("Train Transform Generated")
    print("=" * 60)

    if SYNC_DATA:
        dataset_train = SyntheticDataset(SYNC_DATA_NUMCLASS)
    else:
        dataset_train = MXFaceDataset(DATA_ROOT, train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=per_batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=DROP_LAST,
    )
    NUM_CLASS = train_loader.dataset.classes

    # ======= model & loss & optimizer =======#
    BACKBONE_DICT = {
        "MobileFaceNet": MobileFaceNet,
        "ResNet_50": ResNet_50,
        "ResNet_101": ResNet_101,
        "ResNet_152": ResNet_152,
        "IR_50": IR_50,
        "IR_100": IR_100,
        "IR_101": IR_101,
        "IR_152": IR_152,
        "IR_185": IR_185,
        "IR_200": IR_200,
        "IR_SE_50": IR_SE_50,
        "IR_SE_100": IR_SE_100,
        "IR_SE_101": IR_SE_101,
        "IR_SE_152": IR_SE_152,
        "IR_SE_185": IR_SE_185,
        "IR_SE_200": IR_SE_200,
        "AttentionNet_IR_56": AttentionNet_IR_56,
        "AttentionNet_IRSE_56": AttentionNet_IRSE_56,
        "AttentionNet_IR_92": AttentionNet_IR_92,
        "AttentionNet_IRSE_92": AttentionNet_IRSE_92,
        "ResNeSt_50": resnest50,
        "ResNeSt_101": resnest101,
        "ResNeSt_100": resnest100,
        "GhostNet": GhostNet,
        "MobileNetV3": MobileNetV3,
        "ProxylessNAS": proxylessnas,
        "EfficientNet": efficientnet,
        "DenseNet": densenet,
        "ReXNetV1": ReXNetV1,
        "MobileNeXt": MobileNeXt,
        "MobileNetV2": MobileNetV2,
    }  #'HRNet_W30': HRNet_W30, 'HRNet_W32': HRNet_W32, 'HRNet_W40': HRNet_W40, 'HRNet_W44': HRNet_W44, 'HRNet_W48': HRNet_W48, 'HRNet_W64': HRNet_W64

    BACKBONE_NAME = cfg["BACKBONE_NAME"]
    INPUT_SIZE = cfg["INPUT_SIZE"]
    assert INPUT_SIZE == [112, 112]
    backbone = BACKBONE_DICT[BACKBONE_NAME](INPUT_SIZE)
    print("=" * 60)
    print(backbone)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)
    HEAD_DICT = {
        "Softmax": Softmax,
        "ArcFace": ArcFace,
        "Combined": Combined,
        "CosFace": CosFace,
        "SphereFace": SphereFace,
        "Am_softmax": Am_softmax,
        "CurricularFace": CurricularFace,
        "ArcNegFace": ArcNegFace,
        "SVX": SVXSoftmax,
        "AirFace": AirFace,
        "QAMFace": QAMFace,
        "CircleLoss": CircleLoss,
    }

    HEAD_NAME = cfg["HEAD_NAME"]
    EMBEDDING_SIZE = cfg["EMBEDDING_SIZE"]
    head = HEAD_DICT[HEAD_NAME](in_features=EMBEDDING_SIZE, out_features=NUM_CLASS)
    print("Params: ", count_model_params(backbone))
    print("Flops:", count_model_flops(backbone))
    print("=" * 60)
    print(head)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(backbone)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(
            backbone
        )

    torch.cuda.set_device(cfg["GPU"])
    backbone.cuda(cfg["GPU"])
    head.cuda(cfg["GPU"])

    LR = cfg["LR"]
    WEIGHT_DECAY = cfg["WEIGHT_DECAY"]
    MOMENTUM = cfg["MOMENTUM"]
    params = [
        {
            "params": backbone_paras_wo_bn + list(head.parameters()),
            "weight_decay": WEIGHT_DECAY,
        },
        {"params": backbone_paras_only_bn},
    ]
    if OPTIMIZER == "sgd":
        optimizer = optim.SGD(params, lr=LR, momentum=MOMENTUM)
    elif OPTIMIZER == "adam":
        optimizer = optim.Adam(
            params, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
        )
    elif OPTIMIZER == "lookahead":
        base_optimizer = optim.Adam(
            params, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
        )
        optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)
    elif OPTIMIZER == "radam":
        optimizer = RAdam(params, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif OPTIMIZER == "ranger":
        optimizer = Ranger(params, lr=LR, alpha=0.5, k=6)
    elif OPTIMIZER == "adamp":
        optimizer = AdamP(params, lr=LR, betas=(0.9, 0.999), weight_decay=1e-2)
    elif OPTIMIZER == "sgdp":
        optimizer = SGDP(params, lr=LR, weight_decay=1e-5, momentum=0.9, nesterov=True)

    if LR_SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAT_GAMMA
        )
    elif LR_SCHEDULER == "multi_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_DECAY_EPOCH, gamma=LR_DECAT_GAMMA
        )
    elif LR_SCHEDULER == "cosine":
        scheduler = CosineWarmupLR(
            optimizer,
            batches=len(train_loader),
            epochs=NUM_EPOCH,
            base_lr=LR,
            target_lr=LR_END,
            warmup_epochs=WARMUP_EPOCH,
            warmup_lr=WARMUP_LR,
        )

    print("=" * 60)
    print(optimizer)
    print("Optimizer Generated")
    print("=" * 60)

    # loss
    LOSS_NAME = cfg["LOSS_NAME"]
    LOSS_DICT = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "LabelSmooth": LabelSmoothCrossEntropyLoss(classes=NUM_CLASS),
        "Focal": FocalLoss(),
        "HM": HardMining(),
        "Softplus": nn.Softplus(),
    }
    loss = LOSS_DICT[LOSS_NAME].cuda(gpu)
    print("=" * 60)
    print(loss)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    # optionally resume from a checkpoint
    BACKBONE_RESUME_ROOT = cfg["BACKBONE_RESUME_ROOT"]
    HEAD_RESUME_ROOT = cfg["HEAD_RESUME_ROOT"]
    IS_RESUME = cfg["IS_RESUME"]
    if IS_RESUME:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            loc = "cuda:{}".format(cfg["GPU"])
            backbone.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, map_location=loc))
            if os.path.isfile(HEAD_RESUME_ROOT):
                print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
                checkpoint = torch.load(HEAD_RESUME_ROOT, map_location=loc)
                cfg["START_EPOCH"] = checkpoint["EPOCH"]
                head.load_state_dict(checkpoint["HEAD"])
                optimizer.load_state_dict(checkpoint["OPTIMIZER"])
                del checkpoint
        else:
            print(
                "No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT
                )
            )
        print("=" * 60)
    ori_backbone = copy.deepcopy(backbone)
    if SYNC_BN:
        backbone = apex.parallel.convert_syncbn_model(backbone)
    if USE_AMP:
        [backbone, head], optimizer = amp.initialize(
            [backbone, head], optimizer, opt_level=cfg["OPT_LEVEL"]
        )
        backbone = apex.parallel.DistributedDataParallel(backbone)
        head = apex.parallel.DistributedDataParallel(head)
    else:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[cfg["GPU"]]
        )
        head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[cfg["GPU"]])

    # train
    for epoch in range(cfg["START_EPOCH"], cfg["NUM_EPOCH"]):
        train_sampler.set_epoch(epoch)
        if LR_SCHEDULER != "cosine":
            scheduler.step()
        # train for one epoch
        DISP_FREQ = 100
        batch = 0
        backbone.train()
        head.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for inputs, labels in tqdm(iter(train_loader)):
            if LR_SCHEDULER == "cosine":
                scheduler.step()
            inputs = inputs.cuda(cfg["GPU"], non_blocking=True)
            labels = labels.cuda(cfg["GPU"], non_blocking=True)

            if cfg["MIXUP"]:
                inputs, labels_a, labels_b, lam = mixup_data(
                    inputs, labels, cfg["GPU"], cfg["MIXUP_PROB"], cfg["MIXUP_ALPHA"]
                )
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))
            elif cfg["CUTMIX"]:
                inputs, labels_a, labels_b, lam = cutmix_data(
                    inputs, labels, cfg["GPU"], cfg["CUTMIX_PROB"], cfg["MIXUP_ALPHA"]
                )
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))
            features = backbone(inputs)
            outputs = head(features, labels)

            if cfg["MIXUP"] or cfg["CUTMIX"]:
                lossx = mixup_criterion(loss, outputs, labels_a, labels_b, lam)
            else:
                lossx = (
                    loss(outputs, labels)
                    if HEAD_NAME != "CircleLoss"
                    else loss(outputs).mean()
                )

            optimizer.zero_grad()
            if USE_AMP:
                with amp.scale_loss(lossx, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                lossx.backward()
            optimizer.step()

            prec1, prec5 = (
                accuracy(outputs.data, labels, topk=(1, 5))
                if HEAD_NAME != "CircleLoss"
                else accuracy(features.data, labels, topk=(1, 5))
            )
            losses.update(lossx.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            if ((batch + 1) % DISP_FREQ == 0) or batch == 0:
                print("=" * 60)
                print(
                    "Epoch {}/{} Batch {}/{}\t"
                    "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        epoch + 1,
                        cfg["NUM_EPOCH"],
                        batch + 1,
                        len(train_loader),
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                print("=" * 60)

            if (batch + 1) % EVAL_FREQ == 0:
                # lr = scheduler.get_last_lr()
                lr = optimizer.param_groups[0]["lr"]
                print("Current lr", lr)
                print("=" * 60)
                for vs in val_dataset:
                    result = ver_test(vs, backbone)
                    print(
                        "Epoch {}/{}, Evaluation: {}, Acc: {}, XNorm: {}".format(
                            epoch + 1, NUM_EPOCH, vs[2], result[0], result[2]
                        )
                    )
                print("=" * 60)

                print("=" * 60)
                print("Save Checkpoint...")
                if cfg["RANK"] % ngpus_per_node == 0:
                    """
                    if epoch+1==cfg['NUM_EPOCH']:
                        torch.save(backbone.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, get_time())))
                        save_dict = {'EPOCH': epoch+1,
                                    'HEAD': head.module.state_dict(),
                                    'OPTIMIZER': optimizer.state_dict()}
                        torch.save(save_dict, os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, get_time())))
                    """
                    if USE_AMP:
                        ori_backbone = ori_backbone.half()
                    ori_backbone.load_state_dict(backbone.module.state_dict())
                    ori_backbone.eval()
                    x = torch.randn(1, 3, 112, 112).cuda()
                    traced_cell = torch.jit.trace(ori_backbone, (x))
                    torch.jit.save(
                        traced_cell,
                        os.path.join(
                            MODEL_ROOT,
                            "Epoch_{}_Time_{}_checkpoint.pth".format(
                                epoch + 1, get_time()
                            ),
                        ),
                    )

            sys.stdout.flush()
            batch += 1
        print("=" * 60)
        print(
            "Epoch: {}/{}\t"
            "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
            "Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                epoch + 1, cfg["NUM_EPOCH"], loss=losses, top1=top1, top5=top5
            )
        )
        sys.stdout.flush()
        print("=" * 60)


if __name__ == "__main__":
    main()
