import torch
import random
import numpy as np
from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time():
    return (str(datetime.now())[:-10]).replace(" ", "-").replace(":", "-")


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "resnet" in str(layer.__class__):
            continue
        if "backbone" in str(layer.__class__):
            continue
        if "resattnet" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "batchnorm" in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def separate_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find("bn") >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn
