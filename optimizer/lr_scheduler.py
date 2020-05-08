# -*- coding: utf-8 -*-

__all__ = ['CosineWarmupLR']

from math import pi, cos
from torch.optim.optimizer import Optimizer


class CosineWarmupLR(object):
    """Cosine lr decay function with warmup.

    Lr warmup is proposed by `
        Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`

    Cosine decay is proposed by `
        Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`

    Args:
        optimizer (Optimizer): optimizer of a model.
        batches (int): batches of one epoch.
        epochs (int): epochs to train.
        base_lr (float): init lr.
        target_lr (float): minimum(final) lr.
        warmup_epochs (int): warmup epochs before cosine decay.
        warmup_lr (float): warmup starting lr.
        last_iter (int): init iteration.

    Attributes:
        niters (int): number of iterations of all epochs.
        warmup_iters (int): number of iterations of all warmup epochs.

    """

    def __init__(self, optimizer, batches, epochs, base_lr,
                 target_lr=0, warmup_epochs=0, warmup_lr=0, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.baselr = base_lr
        self.learning_rate = base_lr
        self.niters = epochs * batches
        self.targetlr = target_lr
        self.warmup_iters = batches * warmup_epochs
        self.warmup_lr = warmup_lr
        self.last_iter = last_iter
        #self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            self.learning_rate = self.warmup_lr + (self.baselr - self.warmup_lr) * \
                                 self.last_iter / self.warmup_iters
        else:
            self.learning_rate = self.targetlr + (self.baselr - self.targetlr) * \
                                 (1 + cos(pi * (self.last_iter - self.warmup_iters) /
                                          (self.niters - self.warmup_iters))) / 2

    def step(self, iteration=None):
        """Update status of lr.

        Args:
            iteration(int, optional): now training iteration of all epochs.
                Normally need not to set it manually.
        """
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate