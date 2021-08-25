from backbone.resnet import *
from backbone.resnet_irse import *
from backbone.mobilefacenet import *
from backbone.resattnet import *
from backbone.resnest import *
from backbone.ghostnet import *
from backbone.mobilenetv3 import *
from backbone.proxylessnas import *
from backbone.efficientnet import *
from backbone.densenet import *
from backbone.rexnetv1 import *
from backbone.mobilenext import *
from backbone.mobilenetv2 import *


BACKBONE_DICT = {
    "MobileFaceNet": MobileFaceNet,
    "ResNet_50": ResNet_50,
    "ResNet_101": ResNet_101,
    "ResNet_152": ResNet_152,
    "IR_50": IR_50,
    "IR_100": IR_100,
    "IR_101": IR_101,
    "IR_152": IR_152,
    "IR_185": IR_185,
    "IR_200": IR_200,
    "IR_SE_50": IR_SE_50,
    "IR_SE_100": IR_SE_100,
    "IR_SE_101": IR_SE_101,
    "IR_SE_152": IR_SE_152,
    "IR_SE_185": IR_SE_185,
    "IR_SE_200": IR_SE_200,
    "AttentionNet_IR_56": AttentionNet_IR_56,
    "AttentionNet_IRSE_56": AttentionNet_IRSE_56,
    "AttentionNet_IR_92": AttentionNet_IR_92,
    "AttentionNet_IRSE_92": AttentionNet_IRSE_92,
    "ResNeSt_50": resnest50,
    "ResNeSt_101": resnest101,
    "ResNeSt_100": resnest100,
    "GhostNet": GhostNet,
    "MobileNetV3": MobileNetV3,
    "ProxylessNAS": proxylessnas,
    "EfficientNet": efficientnet,
    "DenseNet": densenet,
    "ReXNetV1": ReXNetV1,
    "MobileNeXt": MobileNeXt,
    "MobileNetV2": MobileNetV2,
}
