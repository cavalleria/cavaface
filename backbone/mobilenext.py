"""
modified from mobilenetv2 torchvision implementation
https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
"""
# ------------------------------------------------------------------------------
# Updated by cavalleria (cavalleria@gmail.com)
# ------------------------------------------------------------------------------


import torch
from torch import nn
from .common import GDC


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.PReLU(out_planes) #nn.ReLU6(inplace=True)
        )


class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=None):
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier==1.0 else True
        self.identity_tensor_channels = int(round(inp*identity_tensor_multiplier))

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp / expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # dw
        layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp, norm_layer=norm_layer))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                norm_layer(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, norm_layer=norm_layer),
            # dw-linear
            nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor= x[:,:self.identity_tensor_channels,:,:] + out[:,:self.identity_tensor_channels,:,:]
                out = torch.cat([identity_tensor, out[:,self.identity_tensor_channels:,:,:]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out


class MobileNeXt(nn.Module):
    def __init__(self,
                input_size, embedding_size = 512, 
                width_mult=1.0, identity_tensor_multiplier=1.0,
                sand_glass_setting=None,
                round_nearest=8,
                block=None,
                norm_layer=None):
        """
        MobileNeXt main class
        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            sand_glass_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNeXt, self).__init__()
        assert input_size[0] in [112]
        if block is None:
            block = SandGlass

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 512

        if sand_glass_setting is None:
            sand_glass_setting = [
                # t, c,  b, s
                #[2, 96,  1, 2],
                [6, 144, 1, 1],
                [6, 192, 3, 2],
                [6, 288, 3, 2],
                [6, 384, 4, 1],
                [6, 576, 4, 2],
                [6, 960, 2, 1],
                [6, last_channel, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(sand_glass_setting) == 0 or len(sand_glass_setting[0]) != 4:
            raise ValueError("sand_glass_setting should be non-empty "
                             "or a 4-element list, got {}".format(sand_glass_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building sand glass blocks
        for t, c, b, s in sand_glass_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(b):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, identity_tensor_multiplier=identity_tensor_multiplier, norm_layer=norm_layer))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.output_layer = GDC(512, embedding_size)
        

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.output_layer(x)
        return x