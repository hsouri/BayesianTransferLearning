
""""""
import warnings
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class ABAnnealingLR(_LRScheduler):
    """Step size scheduler for SGLD.

    a and b are computed based on start and final step size.

    .. math::
      \epsilon_t = a(b + t)^{-\gamma}

    .. _SGLD\: Bayesian Learning via Stochastic Gradient Langevin Dynamics:
          https://icml.cc/2011/papers/398_icmlpaper.pdf
    """

    def __init__(self, optimizer, final_lr, gamma, T_max, last_epoch=-1, verbose=False):
        self.final_lr = final_lr
        self.gamma = gamma
        self.T_max = T_max

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs

        new_lrs = []
        for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups):
            b = self.T_max / ((base_lr / self.final_lr) * math.exp(1 / self.gamma) - 1.)
            a = base_lr * b ** self.gamma

            new_lr = a / (b + self.last_epoch) ** self.gamma
            new_lrs.append(new_lr)

        return new_lrs

    def _get_closed_form_lr(self):
        closed_form_lrs = []
        for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups):
            b = (self.gamma * self.T_max) / (math.log(base_lr / self.final_lr) - 1.)
            a = base_lr * b ** self.gamma

            lr = a / (b + self.last_epoch) ** self.gamma
            closed_form_lrs.append(lr)

        return closed_form_lrs


class CosineLR(_LRScheduler):
    """Cyclic size scheduler for SGLD (a.k.a cSG-MCMC).

    K is the number of total iterations.
    M is the number of cycles.
    beta is the fraction of the cycle for which we do optimization.

    .. math::
      \alpha_k = \frac{\alpha_0}{2} \left[ \cos{\frac{\pi\mod{k-1, \ceil{K/M}}}{\ceil{K/M}}} \right]

    .. _cSG-MCMC\: Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning:
          https://arxiv.org/abs/1902.03932
    """

    def __init__(self, optimizer, n_cycles, n_samples, T_max, beta=1 / 4,
                 last_epoch=-1, verbose=False):
        self.beta = beta
        self._cycle_len = int(math.ceil(T_max / n_cycles))
        self._last_beta = 0.

        samples_per_cycle = n_samples // n_cycles
        self._thres = ((beta + torch.arange(1, samples_per_cycle + 1) * (
                    1 - beta) / samples_per_cycle) * self._cycle_len).int()

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs

        beta = (self.last_epoch % self._cycle_len) / self._cycle_len

        new_lrs = []
        _lr_factor = (math.cos(math.pi * beta) + 1.)
        for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = .5 * base_lr * _lr_factor
            new_lrs.append(new_lr)

        self._last_beta = beta

        return new_lrs

    def get_last_beta(self):
        return self._last_beta

    def _get_closed_form_lr(self):
        beta = (self.last_epoch % self._cycle_len) / self._cycle_len

        closed_form_lrs = []
        _lr_factor = (math.cos(math.pi * beta) + 1.)
        for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups):
            lr = .5 * base_lr * _lr_factor
            closed_form_lrs.append(lr)

        return closed_form_lrs

    def should_sample(self):
        '''Aim for (n_samples // n_cycles) samples per cycle.

        NOTE: Use before the next step() call to scheduler.
        '''

        _t = self.last_epoch % self._cycle_len + 1
        return (_t - self._thres).abs().min() == 0