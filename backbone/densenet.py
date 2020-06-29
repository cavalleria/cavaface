"""
    DenseNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

# ------------------------------------------------------------------------------
# Updated by cavalleria (cavalleria@gmail.com)
# ------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import pre_conv1x1_block, pre_conv3x3_block

class DenseUnit(nn.Module):
    """
    DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate):
        super(DenseUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=inc_channels)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.cat((identity, x), dim=1)
        return x


class TransitionBlock(nn.Module):
    """
    DenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    """
    DenseNet model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
    """
    def __init__(self, channels, init_block_channels, embedding_size=512):
        super(DenseNet, self).__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, init_block_channels, (3, 3), 2, 1, bias=False),
                                    nn.BatchNorm2d(init_block_channels),
                                    nn.PReLU(init_block_channels))
        self.features = nn.Sequential()
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            if i != 0:
                stage.add_module("trans{}".format(i + 1), TransitionBlock(
                    in_channels=in_channels,
                    out_channels=(in_channels // 2)))
                in_channels = in_channels // 2
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), DenseUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=0.0))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.Dropout(0.4),
                                    nn.Flatten(),
                                    nn.Linear(in_channels * 7 * 7, embedding_size),
                                    nn.BatchNorm1d(embedding_size, affine=False))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.features(x)
        x = self.output_layer(x)
        return x


def densenet(input_size, embedding_size=512, blocks=201, **kwargs):
    """
    Create DenseNet model with specific parameters.
    """
    assert input_size[0] in [112]
    if blocks == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif blocks == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif blocks == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif blocks == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported DenseNet version with number of layers {}".format(blocks))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = DenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        embedding_size=embedding_size,
        **kwargs)

    return net


