import torch
import torch.nn as nn
import torchshard as ts


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class HardMining(nn.Module):
    def __init__(self, save_rate=2):
        super(HardMining, self).__init__()
        self.save_rate = save_rate
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        batch_size = input.shape[0]
        loss = self.ce(input, target)
        ind_sorted = torch.argsort(-loss)  # from big to small
        num_saved = int(self.save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss_final = torch.sum(self.ce(input[ind_update], target[ind_update]))
        return loss_final


@torch.no_grad()
def smooth_one_hot(true_labels, classes, smoothing=0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label


class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        loss_final = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return loss_final


class ParallelArcLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super(ParallelArcLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target):
        cos, phi = input
        n, c = cos.shape

        parallel_dim = ts.get_parallel_dim(cos)
        if parallel_dim is not None:
            assert parallel_dim == -1 or parallel_dim == 1

            one_hot = torch.zeros(
                (n, c * ts.distributed.get_world_size()),
                device=cos.device,
                requires_grad=False,
            )
            one_hot.scatter_(1, target.unsqueeze(dim=-1), 1)

            # slice one-hot matirx
            one_hot = ts.distributed.scatter(one_hot, dim=-1)
        else:
            one_hot = torch.zeros((n, c), device=cos.device, requires_grad=False)
            one_hot.scatter_(1, target.unsqueeze(dim=-1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos)

        # set parallel attribute
        ts.register_parallel_dim(output, ts.get_parallel_dim(cos))

        # calculate loss values
        return ts.nn.functional.parallel_cross_entropy(
            output,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
