"""
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
# ------------------------------------------------------------------------------
# Updated by cavalleria (cavalleria@gmail.com)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from math import ceil
from .common import GDC


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def _add_conv(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))
        #out.append(nn.PReLU(channels))


def _add_conv_swish(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(Swish())
    #out.append(nn.PReLU(channels))


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            _add_conv_swish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        _add_conv(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                  num_group=dw_channels, active=False)

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        _add_conv(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x

        return out


class ReXNetV1(nn.Module):
    def __init__(self, input_size, embedding_size = 512, width_mult=1.0, depth_mult=1.0):
        super(ReXNetV1, self).__init__()

        assert input_size[0] in [112]
        input_ch=16; final_ch=180; use_se=True; se_ratio=12
        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        layers = [int(ceil(element * depth_mult)) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        self.depth = sum(layers[:]) * 3

        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        _add_conv_swish(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=1, pad=1)

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        if use_se:
            use_ses = [False] * (layers[0] + layers[1]) + [True] * sum(layers[2:])
        else:
            use_ses = [False] * sum(layers[:])

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_se=se, se_ratio=se_ratio))

        pen_channels = 512 #int(1280 * width_mult)
        _add_conv_swish(features, c, pen_channels)
        self.features = nn.Sequential(*features)
        self.output_layer = GDC(pen_channels, embedding_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.output_layer(x)
        return x