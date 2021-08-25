from loss.loss import *

LOSS_DICT = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    # "LabelSmooth": LabelSmoothCrossEntropyLoss(classes=12),
    "Focal": FocalLoss(),
    "HM": HardMining(),
    "Softplus": nn.Softplus(),
    "ParallelArcLoss": ParallelArcLoss(),
}

