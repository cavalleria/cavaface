import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchshard as ts


class ParallelArcFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dim=None,
        m=0.5,
        s=64,
        easy_margin=True,
    ):
        super(ParallelArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        assert dim != 0, f"ParallelArcFace doesn't support dim = {self.dim}."

        self.m = m  # margin value, default is 0.5
        self.s = s  # scalar value, default is 64.
        self.easy_margin = easy_margin

        self.weight = torch.Tensor(self.out_features, self.in_features)
        self.bias = None  # arcface has no bias

        self.procs_params()
        self.reset_params()
        self.slice_params()

    def procs_params(self):
        # process parameters and other stuff
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.mm = self.sin_m * self.m
        self.threshold = math.cos(math.pi - self.m)

    def reset_params(self):
        self.weight.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def slice_params(self):
        # scatter weight
        if self.dim == -1 or self.dim == 1:
            self.weight = ts.distributed.scatter(self.weight, dim=0)

        if self.dim == 0:
            self.weight = ts.distributed.scatter(self.weight, dim=1)

        # wrap into Parameter
        self.weight = nn.Parameter(self.weight)

        # set parallel attr
        ts.register_parallel_dim(self.weight, self.dim)

    def forward(self, input):
        if self.dim == 1 or self.dim == -1:
            output = _col_parallel_forward(
                input,
                self.weight,
                self.m,
                self.s,
                cos_m=self.cos_m,
                sin_m=self.sin_m,
                mm=self.mm,
                th=self.threshold,
                easy_margin=self.easy_margin,
            )
        elif self.dim == 0:
            raise ValueError(
                f"ParallelArcLinear is not implemented in the row parallel manner."
            )
        else:
            output = _forward(
                input,
                self.weight,
                self.m,
                self.s,
                cos_m=self.cos_m,
                sin_m=self.sin_m,
                mm=self.mm,
                th=self.threshold,
                easy_margin=self.easy_margin,
            )
        return output

    def extra_repr(self):
        return "in_features={}, out_features={}, dim={}, margin={}, scale={}, easy_margin={}".format(
            self.in_features,
            self.out_features,
            self.dim,
            self.m,
            self.s,
            self.easy_margin,
        )


def _col_parallel_forward(input, weight, m, s, cos_m, sin_m, mm, th, easy_margin=True):
    """
    Parallel forward in column dimension.
    """
    input = ts.distributed.copy(input)

    # same as naive forward
    cos, phi = _forward(
        input, weight, m, s, cos_m, sin_m, mm, th, easy_margin=easy_margin
    )

    # set parallel attribute
    ts.register_parallel_dim(cos, -1)
    ts.register_parallel_dim(phi, -1)

    return cos, phi


def _forward(input, weight, m, s, cos_m, sin_m, mm, th, easy_margin=True):
    """
    Naive forward of arcface.
    
    Note:
        Refer to https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py#L35.
    """

    cos = F.linear(F.normalize(input), F.normalize(weight))
    sin = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))

    # torch.where doesn't support fp16 input
    is_half = cos.dtype == torch.float16

    phi = cos * cos_m - sin * sin_m
    if easy_margin:
        phi = torch.where(cos.float() > 0.0, phi.float(), cos.float())
    else:
        phi = torch.where(cos.float() > th, phi.float(), cos.float() - mm)

    if is_half:
        cos = cos.half()
        phi = phi.half()

    cos = s * cos
    phi = s * phi

    return cos, phi
