import torch
import torch.nn as nn


# Support: ['FocalLoss', 'HardMinding']

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
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
    def __init__(self, save_rate = 2):
        super(HardMining, self).__init__()
        self.save_rate = save_rate
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        batch_size = input.shape[0]
        loss = self.ce(input, target)
        ind_sorted = torch.argsort(-loss) # from big to small
        num_saved = int(self.save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss_final = torch.sum(self.ce(input[ind_update], target[ind_update]))
        return loss_final