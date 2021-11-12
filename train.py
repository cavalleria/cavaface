import os
import sys
import torch
import builtins
from apex import amp
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

import torchshard as ts

from config import configurations
from backbone import *
from head import *
from loss import *
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
    # val_dataset = get_val_dataset(cfg["DATA_ROOT"], cfg["VAL_SET"])
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))


def main_worker(gpu, ngpus_per_node, cfg):
    cfg["GPU"] = gpu
    seed = cfg["SEED"]
    set_seed(seed)
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
    if cfg["ENABLE_MODEL_PARALLEL"]:
        # init model parallel processes and groups
        ts.distributed.init_process_group(group_size=cfg["WORLD_SIZE"])

    model_root = cfg["MODEL_ROOT"]
    os.makedirs(model_root, exist_ok=True)

    batch_size = int(cfg["BATCH_SIZE"])
    per_batch_size = int(batch_size / ngpus_per_node)
    workers = int(cfg["NUM_WORKERS"])
    data_root = cfg["DATA_ROOT"]
    sync_data = cfg["SYNC_DATA"]
    sync_data_numclass = cfg["SYNC_DATA_NUMCLASS"]
    optimizer_name = cfg["OPTIMIZER"]
    weight_decay = cfg["WEIGHT_DECAY"]
    momentum = cfg["MOMENTUM"]
    lr = cfg["LR"]
    lr_scheduler = cfg["LR_SCHEDULER"]
    lr_step_size = cfg["LR_STEP_SIZE"]
    lr_decay_epoch = cfg["LR_DECAY_EPOCH"]
    lr_decay_gamma = cfg["LR_DECAT_GAMMA"]
    lr_end = cfg["LR_END"]
    warmup_epoch = cfg["WARMUP_EPOCH"]
    warmup_lr = cfg["WARMUP_LR"]
    num_epoch = cfg["NUM_EPOCH"]
    start_epoch = cfg["START_EPOCH"]
    eval_freq = cfg["EVAL_FREQ"]

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
    transform_list.append(
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
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

    if sync_data:
        dataset_train = SyntheticDataset(sync_data_numclass)
    else:
        dataset_train = MXFaceDataset(data_root, train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=per_batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    num_class = train_loader.dataset.classes

    # ======= model & loss & optimizer =======#
    backbone_name = cfg["BACKBONE_NAME"]
    input_size = cfg["INPUT_SIZE"]
    assert input_size == [112, 112]
    backbone = BACKBONE_DICT[backbone_name](input_size)
    print("=" * 60)
    print(backbone)
    print("{} Backbone Generated".format(backbone_name))
    print("=" * 60)

    head_name = cfg["HEAD_NAME"]
    emd_size = cfg["EMBEDDING_SIZE"]
    if cfg["ENABLE_MODEL_PARALLEL"]:
        head = HEAD_DICT[head_name](
            in_features=emd_size, out_features=num_class, dim=cfg["MODEL_PARALLEL_DIM"]
        )
    else:
        head = HEAD_DICT[head_name](in_features=emd_size, out_features=num_class)
    print("Params: ", count_model_params(backbone))
    print("Flops:", count_model_flops(backbone))
    print("=" * 60)
    print(head)
    print("{} Head Generated".format(head_name))
    print("=" * 60)
    if cfg["ENABLE_MODEL_PARALLEL"] and cfg["MODEL_PARALLEL_DIM"] is not None:
        # let DDP ignore parallel parameters
        ts.register_ddp_parameters_to_ignore(backbone)
        ts.register_ddp_parameters_to_ignore(head)

    if backbone_name.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(backbone)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_bn_paras(backbone)

    torch.cuda.set_device(cfg["GPU"])
    backbone.cuda(cfg["GPU"])
    head.cuda(cfg["GPU"])

    params = [
        {
            "params": backbone_paras_wo_bn + list(head.parameters()),
            "weight_decay": weight_decay,
        },
        {"params": backbone_paras_only_bn},
    ]
    if cfg["ENABLE_ZERO_OPTIM"]:
        optimizer = ZeroRedundancyOptimizer(
            backbone.parameters(),
            optimizer_class=torch.optim.SGD,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        optimizer = get_optimizer(optimizer_name, params, lr, momentum)
    if lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_decay_gamma
        )
    elif lr_scheduler == "multi_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_decay_epoch, gamma=lr_decay_gamma
        )
    elif lr_scheduler == "cosine":
        scheduler = CosineWarmupLR(
            optimizer,
            batches=len(train_loader),
            epochs=num_epoch,
            base_lr=lr,
            target_lr=lr_end,
            warmup_epochs=warmup_epoch,
            warmup_lr=warmup_lr,
        )

    print("=" * 60)
    print(optimizer)
    print("Optimizer Generated")
    print("=" * 60)

    # loss
    loss_name = cfg["LOSS_NAME"]
    loss = LOSS_DICT[loss_name].cuda(gpu)
    print("=" * 60)
    print(loss)
    print("{} Loss Generated".format(loss_name))
    print("=" * 60)

    # ori_backbone = copy.deepcopy(backbone)

    backbone = DDP(backbone, [cfg["GPU"]])
    head = DDP(head, [cfg["GPU"]])
    scaler = GradScaler(enabled=cfg["ENABLE_AMP"])
    # train
    for epoch in range(start_epoch, num_epoch):
        train_sampler.set_epoch(epoch)
        if lr_scheduler != "cosine":
            scheduler.step()
        batch = 0
        backbone.train()
        head.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # gradscaler

        for inputs, labels in tqdm(iter(train_loader)):
            if lr_scheduler == "cosine":
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

            with autocast(enabled=cfg["ENABLE_AMP"]):
                features = backbone(inputs)
                if cfg["ENABLE_MODEL_PARALLEL"]:
                    features = ts.distributed.gather(features, dim=0)
                    outputs = head(features)
                    labels = ts.distributed.gather(labels, dim=0)
                else:
                    outputs = head(features, labels)

                if cfg["MIXUP"] or cfg["CUTMIX"]:
                    lossx = mixup_criterion(loss, outputs, labels_a, labels_b, lam)
                else:
                    lossx = (
                        loss(outputs, labels)
                        if head_name != "CircleLoss"
                        else loss(outputs).mean()
                    )

            if cfg["ENABLE_MODEL_PARALLEL"] and cfg["MODEL_PARALLEL_DIM"] in [1, -1]:
                outputs = outputs[0]
                outputs = ts.distributed.gather(outputs, dim=-1)

            scaler.scale(lossx).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            prec1, prec5 = (
                accuracy(outputs.data, labels, topk=(1, 5))
                if head_name != "CircleLoss"
                else accuracy(features.data, labels, topk=(1, 5))
            )
            losses.update(lossx.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))
            torch.cuda.synchronize()

            if ((batch + 1) % 100 == 0) or batch == 0:
                print("=" * 60)
                print(
                    "Epoch {}/{} Batch {}/{}\t"
                    "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        epoch + 1,
                        num_epoch,
                        batch + 1,
                        len(train_loader),
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                print("=" * 60)

            state_dict = backbone.module.state_dict()
            if cfg["ENABLE_MODEL_PARALLEL"]:
                state_dict = ts.collect_state_dict(backbone, state_dict)
            if (batch + 1) % eval_freq == 0:
                # lr = scheduler.get_last_lr()
                lr = optimizer.param_groups[0]["lr"]
                print("Current lr", lr)
                print("=" * 60)
                for vs in val_dataset:
                    result = ver_test(vs, backbone)
                    print(
                        "Epoch {}/{}, Evaluation: {}, Acc: {}, XNorm: {}".format(
                            epoch + 1, num_epoch, vs[2], result[0], result[2]
                        )
                    )
                print("=" * 60)

                print("=" * 60)
                print("Save Checkpoint...")
                if cfg["RANK"] % ngpus_per_node == 0:
                    if epoch + 1 == num_epoch:
                        torch.save(
                            backbone.module.state_dict(),
                            os.path.join(
                                model_root,
                                "Backbone_{}_Epoch_{}_Time_{}_checkpoint.pth".format(
                                    backbone_name, epoch + 1, get_time()
                                ),
                            ),
                        )

                    """
                    ori_backbone.load_state_dict(backbone.module.state_dict())
                    ori_backbone.eval()
                    x = torch.randn(1, 3, 112, 112).cuda()
                    traced_cell = torch.jit.trace(ori_backbone, (x))
                    torch.jit.save(
                        traced_cell,
                        os.path.join(
                            model_root,
                            "Epoch_{}_Time_{}_checkpoint.pth".format(
                                epoch + 1, get_time()
                            ),
                        ),
                    )
                    """
            sys.stdout.flush()
            batch += 1
        print("=" * 60)
        print(
            "Epoch: {}/{}\t"
            "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
            "Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                epoch + 1, num_epoch, loss=losses, top1=top1, top5=top5
            )
        )
        sys.stdout.flush()
        print("=" * 60)


if __name__ == "__main__":
    main()
