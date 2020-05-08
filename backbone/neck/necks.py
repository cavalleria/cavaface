from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common import *


body_size = [7,7]

class BaseNeck(nn.Module):
    def __init__(self, emb_size, input_channel=512):
        super(BaseNeck, self).__init__()
        self.fc1 = nn.Sequential()

    def forward(self, x):
        return self.fc1(x)

class neck_Z(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_Z, self).__init__(emb_size, input_channel)
        # bn, dropout
        self.fc1 = nn.Sequential(
                nn.BatchNorm2d(input_channel, affine=True),
                nn.Dropout(0.4)
        )

class neck_E(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_E, self).__init__(emb_size, input_channel)
        # bn, dropout, fc, bn
        self.fc1 = nn.Sequential(
                nn.BatchNorm2d(input_channel, affine=True),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(input_channel*body_size[0]*body_size[1], emb_size),
                nn.BatchNorm1d(emb_size, affine=False)
        )

class neck_F(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_F, self).__init__(emb_size, input_channel)
        # bn, dropout, fc
        self.fc1 = nn.Sequential(
                nn.BatchNorm2d(input_channel, affine=True),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(input_channel*body_size[0]*body_size[1], emb_size)
        )

class neck_G(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_G, self).__init__(emb_size, input_channel)
        # bn, fc
        self.fc1 = nn.Sequential(
                nn.BatchNorm2d(input_channel, affine=True),
                Flatten(),
                nn.Linear(input_channel*body_size[0]*body_size[1], emb_size)
        )

class neck_H(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_H, self).__init__(emb_size, input_channel)
        # fc
        self.fc1 = nn.Sequential(
                Flatten(),
                nn.Linear(input_channel*body_size[0]*body_size[1], emb_size)
        )

class neck_I(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_I, self).__init__(emb_size, input_channel)
        # bn, fc, bn
        self.fc1 = nn.Sequential(
                nn.BatchNorm2d(input_channel, affine=True),
                Flatten(),
                nn.Linear(input_channel*body_size[0]*body_size[1], emb_size),
                nn.BatchNorm1d(emb_size, affine=False)
        )

class neck_J(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_J, self).__init__(emb_size, input_channel)
        # fc, bn
        self.fc1 = nn.Sequential(
                Flatten(),
                nn.Linear(input_channel*body_size[0]*body_size[1], emb_size),
                nn.BatchNorm1d(emb_size, affine=False)
        )

class neck_FC(BaseNeck):
    def __init__(self, emb_size, input_channel=512):
        super(neck_FC, self).__init__(emb_size, input_channel)
        # bn, fc, bn
        self.fc1 = nn.Sequential(
                nn.BatchNorm2d(input_channel, affine=True),
                Flatten(),
                nn.Linear(input_channel*body_size[0]*body_size[1], emb_size),
                nn.BatchNorm1d(emb_size, affine=False)
        )

class neck_GAP(BaseNeck):
    def __init__(self, emb_size, input_channel=512, act='prelu'):
        super(neck_GAP, self).__init__(emb_size, input_channel)
        # bn, relu, globalavgpool, flat, fc, bn
        self.fc1 = nn.Sequential(
                nn.BatchNorm2d(input_channel, affine=True),
                get_activation_layer(act),
                nn.AdaptiveAvgPool2d((1,1)),
                Flatten(),
                nn.Linear(input_channel, emb_size),
                nn.BatchNorm1d(emb_size, affine=False)
        )

class neck_GNAP(BaseNeck): # mobilefacenet++
    def __init__(self, emb_size, input_channel=512, act='prelu'):
        super(neck_GNAP, self).__init__(emb_size, input_channel)
        self.filters_in = 512 # param in mobilefacenet
        if emb_size > self.filters_in:
            self.conv1_bn_act = nn.Sequential(
                nn.Conv2d(input_channel, emb_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(emb_size, affine=True),
                get_activation_layer(act)
            )
        else:
            self.conv1_bn_act = nn.Sequential()

        if emb_size < self.filters_in:
            self.bn = nn.BatchNorm2d(self.filters_in, affine=False)
            self.fc1 = nn.Sequential(
                Flatten(),
                nn.BatchNorm1d(self.filters_in, affine=False),
                nn.Linear(self.filters_in, emb_size),
                nn.BatchNorm1d(emb_size, affine=False),
            )
        else:
            self.bn = nn.BatchNorm2d(emb_size, affine=False)
            self.fc1 = nn.Sequential(
                Flatten(),
                nn.BatchNorm1d(emb_size, affine=False)
            )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        if emb_size > self.filters_in:
            self.filters_in = emb_size

    def forward(self, x):
        x = self.conv1_bn_act(x)
        x = self.bn(x)

        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight

        x = self.pool(x)
        return self.fc1(x)

class neck_GDC(BaseNeck): # mobilefacenet_v1
    def __init__(self, emb_size, input_channel=512):
        super(neck_GDC, self).__init__(emb_size, input_channel)
        # bn, relu, globalavgpool, flat, fc, bn
        self.fc1 = nn.Sequential(
                nn.Conv2d(input_channel, input_channel, 7, 1, 0, bias=False, groups=input_channel),
                nn.BatchNorm2d(input_channel, affine=True),
                Flatten(),
                nn.Linear(input_channel, emb_size, bias=False),
                nn.BatchNorm1d(emb_size, affine=False)
        )


supports = ['E', 'F', 'G', 'H', 'I', 'J', 'Z', 'FC', 'GAP', 'GNAP', 'GDC'] 
neck_dict = {s:eval('neck_'+s) for s in supports}

def get_neck(neck_type):
    assert neck_type in supports
    return neck_dict[neck_type]

if __name__=='__main__':
    n = torch.randn(32,512,7,7)
    for s in supports[2:]:
        neck = get_neck(s)(256,512)
        print(s, 'done', type(neck))
        print(neck.fc1)
        neck.forward(n)
