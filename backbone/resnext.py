"""ResNeXt models"""

from .resnet import ResNet, Bottleneck

__all__ = ['resnext50_32x4d', 'resnext101_32x8d']

def resnext50_32x4d(input_size, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['bottleneck_width'] = 4
    model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnext101_32x8d(input_size, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['bottleneck_width'] = 8
    model = ResNet(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

