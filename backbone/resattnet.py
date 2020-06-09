import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
from .common import Flatten, l2_norm, SEModule, bottleneck_IR, bottleneck_IR_SE

# Support: ['AttentionNet_IR_56', 'AttentionNet_IRSE_56', 'AttentionNet_IR_92', 'AttentionNet_IRSE_92']


class AttentionModule_stage1(nn.Module):

    # input size is 56*56
    def __init__(self, in_channel, out_channel, mode='ir', size1=(56, 56), size2=(28, 28), size3=(14, 14)):
        super(AttentionModule_stage1, self).__init__()
        if mode == 'ir':
            ResidualBlock = bottleneck_IR
        elif mode == 'ir_se':
            ResidualBlock = bottleneck_IR_SE
        self.share_residual_block = ResidualBlock(in_channel, out_channel)
        self.trunk_branches = nn.Sequential(ResidualBlock(in_channel, out_channel),
                                            ResidualBlock(in_channel, out_channel))

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block1 = ResidualBlock(in_channel, out_channel)
        self.skip_connect1 = ResidualBlock(in_channel, out_channel)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block2 = ResidualBlock(in_channel, out_channel)
        self.skip_connect2 = ResidualBlock(in_channel, out_channel)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block3 = nn.Sequential(ResidualBlock(in_channel, out_channel),
                                         ResidualBlock(in_channel, out_channel))

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.mask_block4 = ResidualBlock(in_channel, out_channel)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.mask_block5 = ResidualBlock(in_channel, out_channel)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.mask_block6 = nn.Sequential(nn.BatchNorm2d(out_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channel, out_channel, 1, 1, bias=False),
                                        nn.BatchNorm2d(out_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channel, out_channel, 1, 1, bias=False),
                                        nn.Sigmoid()
        )
        self.last_block = ResidualBlock(in_channel, out_channel)

    def forward(self, x):
        x = self.share_residual_block(x)
        out_trunk = self.trunk_branches(x)

        out_pool1 = self.mpool1(x)
        out_block1 = self.mask_block1(out_pool1)
        out_skip_connect1 = self.skip_connect1(out_block1)

        out_pool2 = self.mpool2(out_block1)
        out_block2 = self.mask_block2(out_pool2)
        out_skip_connect2 = self.skip_connect2(out_block2)

        out_pool3 = self.mpool3(out_block2)
        out_block3 = self.mask_block3(out_pool3)
        
        out_inter3 = self.interpolation3(out_block3) + out_block2
        out = out_inter3 + out_skip_connect2
        out_block4 = self.mask_block4(out)

        out_inter2 = self.interpolation2(out_block4) + out_block1
        out = out_inter2 + out_skip_connect1
        out_block5 = self.mask_block5(out)

        out_inter1 = self.interpolation1(out_block5) + out_trunk
        out_block6 = self.mask_block6(out_inter1)

        out = (1 + out_block6) + out_trunk
        out_last = self.last_block(out)

        return out_last

class AttentionModule_stage2(nn.Module):

    # input image size is 28*28
    def __init__(self, in_channels, out_channels, mode='ir', size1=(28, 28), size2=(14, 14)):
        super(AttentionModule_stage2, self).__init__()
        if mode == 'ir':
            ResidualBlock = bottleneck_IR
        elif mode == 'ir_se':
            ResidualBlock = bottleneck_IR_SE
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax4_blocks = nn.Sequential(nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
        out = out_interp2 + out_skip1_connection

        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax3) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule_stage3(nn.Module):

    # input image size is 14*14
    def __init__(self, in_channels, out_channels, mode='ir', size1=(14, 14)):
        super(AttentionModule_stage3, self).__init__()
        if mode == 'ir':
            ResidualBlock = bottleneck_IR
        elif mode == 'ir_se':
            ResidualBlock = bottleneck_IR_SE
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2_blocks = nn.Sequential(nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class Backbone_56(nn.Module):
    # for input size 112
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone_56, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers == 56, "num_layers should be 56"
        if mode == 'ir':
            ResidualBlock = bottleneck_IR
        elif mode == 'ir_se':
            ResidualBlock = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.residual_block1 = ResidualBlock(64, 64, 2)
        self.attention_module1 = AttentionModule_stage1(64, 64, mode)
        self.residual_block2 = ResidualBlock(64, 128, 2)
        self.attention_module2 = AttentionModule_stage2(128, 128, mode)
        self.residual_block3 = ResidualBlock(128, 256, 2)
        self.attention_module3 = AttentionModule_stage3(256, 256, mode)
        self.residual_block4 = ResidualBlock(256, 512, 2)
        self.residual_block5 = ResidualBlock(512, 512)
        self.residual_block6 = ResidualBlock(512, 512)
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=False))
        self._initialize_weights()
    def forward(self, x):
        out = self.input_layer(x)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        conv_out = self.residual_block6(out)
        out = self.output_layer(conv_out)

        return out

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

class Backbone_92(nn.Module):
    # for input size 112
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone_92, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers == 92, "num_layers should be 92"
        if mode == 'ir':
            ResidualBlock = bottleneck_IR
        elif mode == 'ir_se':
            ResidualBlock = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.residual_block1 = ResidualBlock(64, 64, 2)
        self.attention_module1 = AttentionModule_stage1(64, 64, mode)
        self.residual_block2 = ResidualBlock(64, 128, 2)
        self.attention_module2   = AttentionModule_stage2(128, 128, mode)
        self.attention_module2_2 = AttentionModule_stage2(128, 128, mode)
        self.residual_block3 = ResidualBlock(128, 256, 2)
        self.attention_module3   = AttentionModule_stage3(256, 256, mode)
        self.attention_module3_2 = AttentionModule_stage3(256, 256, mode)
        self.attention_module3_3 = AttentionModule_stage3(256, 256, mode)
        self.residual_block4 = ResidualBlock(256, 512, 2)
        self.residual_block5 = ResidualBlock(512, 512)
        self.residual_block6 = ResidualBlock(512, 512)
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=False))
        self._initialize_weights()
    
    def forward(self, x):
        out = self.input_layer(x)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        conv_out = self.residual_block6(out)
        out = self.output_layer(conv_out)
        return out

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
        

def AttentionNet_IR_56(input_size=[112,112]):
    """Constructs a AttentionNet_IR_56 model.
    """
    model = Backbone_56(input_size, 56, 'ir')
    return model

def AttentionNet_IRSE_56(input_size=[112,112]):
    """Constructs a AttentionNet_IRSE_56 model.
    """
    model = Backbone_56(input_size, 56, 'ir_se')
    return model

def AttentionNet_IR_92(input_size=[112,112]):
    """Constructs a AttentionNet_IR_92 model.
    """
    model = Backbone_92(input_size, 92, 'ir')
    return model

def AttentionNet_IRSE_92(input_size=[112,112]):
    """Constructs a AttentionNet_IRSE_92 model.
    """
    model = Backbone_92(input_size, 92, 'ir_se')
    return model
