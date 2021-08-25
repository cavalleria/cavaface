
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils.io import GPU_MEM_MAX, get_min_free_gpu_mem
import commands
from .pytorch_memlab import MemReporter

def estimate_unit_mem(model, shape, ctx):
    model.eval()
    reporter = MemReporter(model)
    img = torch.from_numpy(np.ones([1]+shape,dtype=np.float32)).to(ctx)
    out = model(img)[0]
    unitmem = reporter.report(verbose=True)
    return unitmem

def estimate_batch_size(gpuid, unitmem):
    mm = get_min_free_gpu_mem([gpuid])
    batch_size = int(mm // unitmem)
    return batch_size


def forward(net, blob):
    tensor = torch.from_numpy(blob.astype(np.float32)).to(net.ctx)
    tensor = (tensor-127.5) / 128.0
    output = net.model(tensor)
    res = []
    for o in output:
        res.append(o.cpu().detach().numpy())
    return res



