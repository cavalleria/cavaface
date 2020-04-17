# -*- coding: utf-8 -*-
"""
EfficientPolyFace
"""
import torch
import math
import numpy as np
import torch.nn as nn
import sys
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import interpolate
import cv2

class get_fc_E(nn.Module):
    def __init__(self, BN, in_feature, in_h, in_w, out_feature):
        super(get_fc_E, self).__init__()
        self.bn1 = BN(in_feature, affine=True, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
        self.bn2 = nn.BatchNorm1d(out_feature, affine=False, eps=2e-5, momentum=0.9)
    def forward(self, x):
        output = {}
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output['feature_nobn'] = x
        x = self.bn2(x)
        output['feature'] = x
        return output

__all__ = ['apolynet_stodepth', 'apolynet_stodepth_deep', 'apolynet_stodepth_deeper']

BN = None

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn = BN(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Step1(nn.Module):

    def __init__(self):
        super(Step1,self).__init__()
        self.stem = nn.Sequential(
            BasicConv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch0 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)
        self.branch1 = BasicConv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        xs = self.stem(x)
        x0 = self.branch0(xs)
        x1 = self.branch1(xs)
        out = torch.cat((x0, x1), 1)
        return out

class Step2(nn.Module):

    def __init__(self):
        super(Step2, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 160, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = 0)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 160, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 64, out_channels = 64, kernel_size = (7,1), stride = (1,1), padding = (3,0)),
            BasicConv2d(in_channels = 64, out_channels = 64, kernel_size = (1,7), stride = (1,1), padding = (0,3)),
            BasicConv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = 0)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Step3(nn.Module):

    def __init__(self):
        super(Step3, self).__init__()
        self.branch0 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)
        self.branch1 = BasicConv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.step1 = Step1()
        self.step2 = Step2()
        self.step3 = Step3()

    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        return x


class BlockA(nn.Module):

    def __init__(self, use_checkpoint=False, keep_prob=0.8, multFlag=True):
        super(BlockA, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch2 = BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 384, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN(384)
        )
        self.prob = keep_prob
        self.multFlag = multFlag
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        a = torch.equal(self.m.sample(),torch.ones(1))
        identity = x.clone()
        if self.training:
            if a:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch2.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                if self.use_checkpoint:
                    x0 = checkpoint(self.branch0, x)
                    x1 = checkpoint(self.branch1, x)
                    x2 = checkpoint(self.branch2, x)
                    out = torch.cat((x0, x1, x2), 1)
                    out = checkpoint(self.stem, out)
                else:
                    x0 = self.branch0(x)
                    x1 = self.branch1(x)
                    x2 = self.branch2(x)
                    out = torch.cat((x0, x1, x2), 1)
                    out = self.stem(out)
                result = identity + 0.3*out
            else:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch2.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False
                result = identity
        else:
            if self.use_checkpoint:
                x0 = checkpoint(self.branch0, x)
                x1 = checkpoint(self.branch1, x)
                x2 = checkpoint(self.branch2, x)
                out = torch.cat((x0, x1, x2), 1)
                out = checkpoint(self.stem, out)
            else:
                x0 = self.branch0(x)
                x1 = self.branch1(x)
                x2 = self.branch2(x)
                out = torch.cat((x0, x1, x2), 1)
                out = self.stem(out)
            if self.multFlag:
                result = identity + 0.3 * out * self.prob
            else:
                retult = identity + 0.3 * out

        result = self.relu(result)
        return result

class BlockABranch(nn.Module):
    def __init__(self):
        super(BlockABranch, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch2 = BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 384, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN(384)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.stem(out)
        return out

class BlockA2B(nn.Module):

    def __init__(self):
        super(BlockA2B, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch1 = BasicConv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 2, padding = 0)
        self.branch2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class BlockB(nn.Module):

    def __init__(self, use_checkpoint=False, keep_prob=0.8, multFlag=True):
        super(BlockB, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 128, out_channels = 160, kernel_size = (1,7), stride = (1,1), padding = (0,3)),
            BasicConv2d(in_channels = 160, out_channels = 192, kernel_size = (7,1), stride = (1,1), padding = (3,0))
        )
        self.branch1 = BasicConv2d(in_channels = 1152, out_channels = 192, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 1152, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN(1152)
        )
        self.prob = keep_prob
        self.multFlag = multFlag
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        a = torch.equal(self.m.sample(),torch.ones(1))
        identity = x.clone()
        if self.training:
            if a:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                if self.use_checkpoint:
                    x0 = checkpoint(self.branch0, x)
                    x1 = checkpoint(self.branch1, x)
                    out = torch.cat((x0, x1), 1)
                    out = checkpoint(self.stem, out)
                else:
                    x0 = self.branch0(x)
                    x1 = self.branch1(x)
                    out = torch.cat((x0, x1), 1)
                    out = self.stem(out)
                result = identity + 0.3*out
            else:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False
                result = identity
        else:
            if self.use_checkpoint:
                x0 = checkpoint(self.branch0, x)
                x1 = checkpoint(self.branch1, x)
                out = torch.cat((x0, x1), 1)
                out = checkpoint(self.stem, out)
            else:
                x0 = self.branch0(x)
                x1 = self.branch1(x)
                out = torch.cat((x0, x1), 1)
                out = self.stem(out)
            if self.multFlag:
                result = identity + 0.3 * out * self.prob
            else:
                retult = identity + 0.3 * out

        result = self.relu(result)
        return result

class BlockBBranch(nn.Module):
    def __init__(self):
        super(BlockBBranch, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 128, out_channels = 160, kernel_size = (1,7), stride = (1,1), padding = (0,3)),
            BasicConv2d(in_channels = 160, out_channels = 192, kernel_size = (7,1), stride = (1,1), padding = (3,0))
        )
        self.branch1 = BasicConv2d(in_channels = 1152, out_channels = 192, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 1152, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN(1152)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.stem(out)

        return out

class BlockB2C(nn.Module):

    def __init__(self):
        super(BlockB2C, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out #2048

class BlockC(nn.Module):

    def __init__(self, use_checkpoint=False, keep_prob=0.8, multFlag=True):
        super(BlockC, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 2048, out_channels = 192, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 192, out_channels = 224, kernel_size = (1,3), stride = (1,1), padding = (0,1)),
            BasicConv2d(in_channels = 224, out_channels = 256, kernel_size = (3,1), stride = (1,1), padding = (1,0))
        )
        self.branch1 = BasicConv2d(in_channels = 2048, out_channels = 192, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 2048, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN(2048)
        )
        self.prob = keep_prob
        self.multFlag = multFlag
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        a = torch.equal(self.m.sample(),torch.ones(1))
        identity = x.clone()
        if self.training:
            if a:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                if self.use_checkpoint:
                    x0 = checkpoint(self.branch0, x)
                    x1 = checkpoint(self.branch1, x)
                    out = torch.cat((x0, x1), 1)
                    out = checkpoint(self.stem, out)
                else:
                    x0 = self.branch0(x)
                    x1 = self.branch1(x)
                    out = torch.cat((x0, x1), 1)
                    out = self.stem(out)
                result = identity + 0.3*out
            else:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False
                result = identity
        else:
            if self.use_checkpoint:
                x0 = checkpoint(self.branch0, x)
                x1 = checkpoint(self.branch1, x)
                out = torch.cat((x0, x1), 1)
                out = checkpoint(self.stem, out)
            else:
                x0 = self.branch0(x)
                x1 = self.branch1(x)
                out = torch.cat((x0, x1), 1)
                out = self.stem(out)
            if self.multFlag:
                result = identity + 0.3 * out * self.prob
            else:
                retult = identity + 0.3 * out

        result = self.relu(result)
        return result

class APolynet(nn.Module):

    def __init__(self, feature_dim, bn_mom=0.1, bn_eps=1e-10, fc_type='E',
                 num_blocks=[10, 20, 10],
                 checkpoints=[0, 0, 0],
                 att_mode='none'):
        super(APolynet, self).__init__()

        self.att_mode = att_mode

        global BN
        def BNFunc(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, momentum=bn_mom, eps=bn_eps)
        BN = BNFunc

        self.stem = Stem()

        self.a2b = BlockA2B()
        self.b2c = BlockB2C()

        if self.att_mode == 'none':
            self.a10 = self._make_layer(BlockA, num_blocks[0], checkpoints=checkpoints[0])
            self.b20 = self._make_layer(BlockB, num_blocks[1], checkpoints=checkpoints[1])
            self.c10 = self._make_layer(BlockC, num_blocks[2], checkpoints=checkpoints[2])
        else:
            raise RuntimeError('unknown att_mode: {}'.format(self.att_mode))

        BN2 = nn.BatchNorm2d

        self.fc = get_fc_E(BN2, 2048, 6, 6, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                scale = math.sqrt(3. / fan_in)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()


    def _make_layer(self, block, num_blocks, checkpoints=0):
        layers = []
        for i in range(num_blocks):
            layers.append(block(use_checkpoint=checkpoints>i))

        return nn.Sequential(*layers)

    def forward(self, x, flip=False):
        output = {}

        img_list = []
        for cnt in range(x.size(0)):
            tmp = x[cnt]
            tmp = tmp.cpu().numpy()
            tmp = tmp.astype(np.uint8)
            tmp = tmp.transpose((1, 2, 0))
            tmp = cv2.resize(tmp[1:, 1:, :], (235, 235))
            tmp = tmp*3.2/255.0 - 1.6
            if flip:
                tmp = cv2.flip(tmp, 1)
            tmp = tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp)
            tmp = tmp.float()
            tmp = tmp[None, ...]
            img_list.append(tmp)
        x = torch.cat(img_list, 0)
        x = x.cuda()

        ori = x
        x = self.stem(x)

        if self.att_mode == 'none':
            x = self.a10(x)
            x = self.a2b(x)
            x = self.b20(x)
            x = self.b2c(x)
            x = self.c10(x)
        else:
            raise RuntimeError('unknown att_mode: {}'.format(self.att_mode))

        headout= self.fc(x)
        output.update(headout)
        return output

def apolynet_stodepth(feature_dim, **kwargs):
    model = APolynet(feature_dim, **kwargs)
    return model

def apolynet_stodepth_deep(feature_dim, **kwargs):
    model = APolynet(feature_dim, num_blocks=[20, 30, 20], **kwargs)
    return model

def apolynet_stodepth_deeper(feature_dim, **kwargs):
    model = APolynet(feature_dim, num_blocks=[23, 38, 23], **kwargs)
    return model