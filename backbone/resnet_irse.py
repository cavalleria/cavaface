import torch
import torch.nn as nn
from torch.nn import (
    Linear,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    Dropout,
    Sequential,
    Module,
)
from collections import namedtuple
from .common import Flatten, bottleneck_IR, bottleneck_IR_SE
from config import configurations

cfg = configurations[1]


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 101:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=23),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 185:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=22),
            get_block(in_channel=128, depth=256, num_units=33),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=24),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode="ir"):
        super(Backbone, self).__init__()
        assert input_size[0] in [
            112,
            224,
        ], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [
            50,
            100,
            101,
            152,
            185,
            200,
        ], "num_layers should be 50, 100, 152, 185, 200"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        """
        
        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(0.4, inplace=True),
                Flatten(),
                Linear(512 * 7 * 7, 512),
                BatchNorm1d(512, affine=False),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(0.4, inplace=True),
                Flatten(),
                Linear(512 * 14 * 14, 512),
                BatchNorm1d(512, affine=False),
            )
        """
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05,)
        self.dropout = nn.Dropout(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        with torch.cuda.amp.autocast(cfg["ENABLE_AMP"]):
            x = self.input_layer(x)
            x = self.body(x)
            # x = self.output_layer(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if cfg["ENABLE_AMP"] else x)
        x = self.features(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, "ir")

    return model


def IR_100(input_size):
    """Constructs a ir-100 model.
    """
    model = Backbone(input_size, 100, "ir")

    return model


def IR_101(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 101, "ir")

    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, "ir")

    return model


def IR_185(input_size):
    """Constructs a ir-185 model.
    """
    model = Backbone(input_size, 185, "ir")

    return model


def IR_200(input_size):
    """Constructs a ir-200 model.
    """
    model = Backbone(input_size, 200, "ir")

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, "ir_se")

    return model


def IR_SE_100(input_size):
    """Constructs a ir_se-100 model.
    """
    model = Backbone(input_size, 100, "ir_se")

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 101, "ir_se")

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, "ir_se")

    return model


def IR_SE_185(input_size):
    """Constructs a ir_se-185 model.
    """
    model = Backbone(input_size, 185, "ir_se")

    return model


def IR_SE_200(input_size):
    """Constructs a ir_se-200 model.
    """
    model = Backbone(input_size, 200, "ir_se")

    return model

