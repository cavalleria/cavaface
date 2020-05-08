
import torch
import numpy as np
import numbers
from torch.nn import functional as F

@torch.no_grad()
def smooth_one_hot(true_labels, classes, smoothing=0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label