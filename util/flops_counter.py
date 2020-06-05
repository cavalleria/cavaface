# -*- coding:utf8 -*-
import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable

import numpy as np

def flops_to_string(flops, units='GFLOPS', precision=2):
    if units == 'GFLOPS':
        return str(round(flops / 10.**9, precision)) + ' ' + units
    elif units == 'MFLOPS':
        return str(round(flops / 10.**6, precision)) + ' ' + units
    elif units == 'KFLOPS':
        return str(round(flops / 10.**3, precision)) + ' ' + units
    else:
        return str(flops) + ' FLOPS'


def params_to_string(params_num, units='M', precision=2):
    if units == 'M':
        return str(round(params_num / 10.**6, precision)) + ' ' + units
    elif units == 'K':
        return str(round(params_num / 10.**3, precision)) + ' ' + units
    else:
        return str(params_num)

def count_model_params(model):
    total_params = np.sum([param.numel() if param.requires_grad else 0 for param in model.parameters()])
    
    return params_to_string(total_params)


def count_model_flops(model, input_res=[112, 112], multiply_adds=True):
    
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        if self.bias is not None:
            bias_ops = self.bias.nelement() if self.bias.nelement() else 0
            flops = batch_size * (weight_ops + bias_ops)
        else:
            flops = batch_size * weight_ops
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    handles = []
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                handles.append(net.register_forward_hook(linear_hook))
            if isinstance(net, torch.nn.BatchNorm2d):
                handles.append(net.register_forward_hook(bn_hook))
            if isinstance(net, torch.nn.ReLU) or isinstance(net, torch.nn.PReLU):
                handles.append(net.register_forward_hook(relu_hook))
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                handles.append(net.register_forward_hook(pooling_hook))
            return
        for c in childrens:
            foo(c)
    
    model.eval()
    foo(model)
    input = Variable(torch.rand(3,input_res[1],input_res[0]).unsqueeze(0), requires_grad = True)
    out = model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    for h in handles:
        h.remove()
    model.train()
    return flops_to_string(total_flops)