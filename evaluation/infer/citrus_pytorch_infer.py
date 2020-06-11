#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.pytorch_tool import estimate_unit_mem, estimate_batch_size, forward
from citrus_base_infer import *
import time

reload(sys)
sys.setdefaultencoding('utf-8')

class CitrusPytorchInfer(CitrusBaseInfer):
    def __init__(self, args, dtype='fp32'):
        super(CitrusPytorchInfer, self).__init__(args=args, dtype=dtype)

    def _load_model(self):
        unitmem = None
        self._image_shape = [int(x) for x in self._args.image_size.split(',')]
        self.fwd_func = forward
        assert self._args.net_scale is not None, "must set net_scale, light/large"
        self._net_scale_type = self._args.net_scale

        self._batch_size = 0

        spath = self._args.model_path.strip().split(',')
        for g in self._args.gpus.split(','):
            g = int(g)
            tnet = edict()
            tnet.ctx = 'cuda:{}'.format(g)
            tnet.model=torch.jit.load(spath[0], map_location=tnet.ctx)
            tnet.model.eval()
            if unitmem is None:
                unitmem = estimate_unit_mem(tnet.model, self._image_shape, tnet.ctx) 
            if self._args.net_scale == 'light':
                tnet.batch_size = estimate_batch_size(g, unitmem)
            else:
                tnet.batch_size = 3*estimate_batch_size(g, unitmem)
            tnet.gpu = g
            self._nets.append(tnet)

            self._batch_size += tnet.batch_size


        print("Finish loading model %s, "
                "infer with shape: %s" %(spath[0], (self._batch_size, 3,
                self._image_shape[1], self._image_shape[2])))
    #


if __name__ == "__main__":
    a = edict()
    a.gpus="1"
    a.model_path="model.pth"
    a.model_type = "mxnet_fp32"
    a.image_size="3,112,112"
    cc = CitrusPytorchInfer(a)
    _face_scrub_list = "../data/citrus_self_warpv2/probe_112/probe_112.lst"
    _face_scrub_root = "../data/citrus_self_warpv2/probe_112"
    _face_scrub = open(_face_scrub_list).readlines()

    imgs_list = []
    for idx, line in enumerate(_face_scrub):
       sline = line.strip().split('\t')
       image_path = sline[0].strip()
       class_id = sline[1].strip()
       out_path = os.path.join("temp/", image_path+".bin")
       image_path = os.path.join(_face_scrub_root, image_path)
       item = (image_path, out_path)
       imgs_list.append(item)
    
    from utils.io import _read_image, _write_bin
    cc.infer_embedding(imgs_list, read_func = _read_image, write_func=_write_bin, is_flip=True)
    print("ccccc")


