"""
    Common routines for models in PyTorch.
"""

__all__ = ['round_channels', 'Identity', 'Swish', 'HSigmoid', 'HSwish', 'get_activation_layer', 'InterpolationBlock',
           'IBN', 'Flatten', 'l2_norm', 'SEModule', 'bottleneck_IR', 'bottleneck_IR_SE']

import math
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple

def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).

    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels

class Identity(nn.Module):
    """
    Identity block.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.

    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "hsigmoid":
            return HSigmoid()
        elif activation == "identity":
            return Identity()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation

class InterpolationBlock(nn.Module):
    """
    Interpolation upsampling block.

    Parameters:
    ----------
    scale_factor : float
        Multiplier for spatial size.
    mode : str, default 'bilinear'
        Algorithm used for upsampling.
    align_corners : bool, default True
        Whether to align the corner pixels of the input and output tensors.
    """
    def __init__(self,
                 scale_factor,
                 mode="bilinear",
                 align_corners=True):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            input=x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)

    def __repr__(self):
        s = "{name}(scale_factor={scale_factor}, mode={mode}, align_corners={align_corners})"
        return s.format(
            name=self.__class__.__name__,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        if self.mode == "bilinear":
            num_flops = 9 * x.numel()
        else:
            num_flops = 4 * x.numel()
        num_macs = 0
        return num_flops, num_macs

class IBN(nn.Module):
    """
    Instance-Batch Normalization block from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    channels : int
        Number of channels.
    inst_fraction : float, default 0.5
        The first fraction of channels for normalization.
    inst_first : bool, default True
        Whether instance normalization be on the first part of channels.
    """
    def __init__(self,
                 channels,
                 first_fraction=0.5,
                 inst_first=True):
        super(IBN, self).__init__()
        self.inst_first = inst_first
        h1_channels = int(math.floor(channels * first_fraction))
        h2_channels = channels - h1_channels
        self.split_sections = [h1_channels, h2_channels]

        if self.inst_first:
            self.inst_norm = nn.InstanceNorm2d(
                num_features=h1_channels,
                affine=True)
            self.batch_norm = nn.BatchNorm2d(num_features=h2_channels)
        else:
            self.batch_norm = nn.BatchNorm2d(num_features=h1_channels)
            self.inst_norm = nn.InstanceNorm2d(
                num_features=h2_channels,
                affine=True)

    def forward(self, x):
        x1, x2 = torch.split(x, split_size_or_sections=self.split_sections, dim=1)
        if self.inst_first:
            x1 = self.inst_norm(x1.contiguous())
            x2 = self.batch_norm(x2.contiguous())
        else:
            x1 = self.batch_norm(x1.contiguous())
            x2 = self.inst_norm(x2.contiguous())
        x = torch.cat((x1, x2), dim=1)
        return x

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride=1):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), 
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), 
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride=1):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

from torch.nn.modules.utils import _pair
class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1) #https://github.com/pytorch/pytorch/pull/32493/files
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size, share_channel=False):
        super(DropBlock2D, self).__init__()
        self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
        self.register_buffer('drop_prob', drop_prob * torch.ones(1, dtype=torch.float32))
        self.inited = False
        self.step_size = 0.0
        self.start_step = 0
        self.nr_steps = 0
        self.block_size = block_size
        self.share_channel = share_channel

    def reset(self):
        """stop DropBlock"""
        self.inited = True
        self.i[0] = 0
        self.drop_prob = 0.0

    def reset_steps(self, start_step, nr_steps, start_value=0, stop_value=None):
        self.inited = True
        stop_value = self.drop_prob.item() if stop_value is None else stop_value
        self.i[0] = 0
        self.drop_prob[0] = start_value
        self.step_size = (stop_value - start_value) / nr_steps
        self.nr_steps = nr_steps
        self.start_step = start_step

    def forward(self, x):
        if not self.training or self.drop_prob.item() == 0.:
            return x
        else:
            self.step()

            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask and place on input device
            if self.share_channel:
                mask = (torch.rand(*x.shape[2:], device=x.device, dtype=x.dtype) < gamma).unsqueeze(0).unsqueeze(0)
            else:
                mask = (torch.rand(*x.shape[1:], device=x.device, dtype=x.dtype) < gamma).unsqueeze(0)

            # compute block mask
            block_mask, keeped = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * (block_mask.numel() / keeped).to(out)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        keeped = block_mask.numel() - block_mask.sum().to(torch.float32)
        block_mask = 1 - block_mask

        return block_mask, keeped

    def _compute_gamma(self, x):
        _, c, h, w = x.size()
        gamma = self.drop_prob.item() / (self.block_size ** 2) * (h * w) / \
            ((w - self.block_size + 1) * (h - self.block_size + 1))
        return gamma

    def step(self):
        assert self.inited
        idx = self.i.item()
        if idx > self.start_step and idx < self.start_step + self.nr_steps:
            self.drop_prob += self.step_size
        self.i += 1

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        idx_key = prefix + 'i'
        drop_prob_key = prefix + 'drop_prob'
        if idx_key not in state_dict:
            state_dict[idx_key] =  torch.zeros(1, dtype=torch.int64)
        if idx_key not in drop_prob_key:
            state_dict[drop_prob_key] =  torch.ones(1, dtype=torch.float32)
        super(DropBlock2D, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """overwrite save method"""
        pass

    def extra_repr(self):
        return 'drop_prob={}, step_size={}'.format(self.drop_prob, self.step_size)

def reset_dropblock(start_step, nr_steps, start_value, stop_value, m):
    """
    Example:
        from functools import partial
        apply_drop_prob = partial(reset_dropblock, 0, epochs*iters_per_epoch, 0.0, 0.1)
        net.apply(apply_drop_prob)
    """
    if isinstance(m, DropBlock2D):
        m.reset_steps(start_step, nr_steps, start_value, stop_value)


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GDC(nn.Module):
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(in_c, in_c, groups=in_c, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(in_c, embedding_size, bias=False)
        #self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x