import math
import torch
from torch import Tensor
from torch.optim import SGD
from typing import List


def sgld(params: List[Tensor],
         d_p_list: List[Tensor],
         momentum_buffer_list: List[Tensor],
         *,
         weight_decay: float,
         lr: float,
         momentum: float,
         noise: bool,
         temperature: float):
    r"""Functional API for SGMCMC/SGHMC.

    .. _SGLD\: Bayesian Learning via Stochastic Gradient Langevin Dynamics:
          https://icml.cc/2011/papers/398_icmlpaper.pdf
    .. _SGHMC\: Stochastic Gradient Hamiltonian Monte Carlo:
          http://www.istc-cc.cmu.edu/publications/papers/2014/Guestrin-stochastic-gradient.pdf
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p.add_(param, alpha=weight_decay)

        sqrt_lr = math.sqrt(lr)
        # sqrt_lr = lr
        noise_std = math.sqrt(2 * (momentum) * temperature)

        buf = momentum_buffer_list[i]
        buf.mul_(1-momentum).add_(d_p, alpha=-(sqrt_lr ** 2))
        if noise:
            eps = torch.randn_like(d_p)
            buf.add_(eps, alpha=sqrt_lr * noise_std)

        param.add_(buf)
        # param.add_(buf, alpha = sqrt_lr)


class SGLD(SGD):
    """Implements SGLD/SGHMC updates.

    Assumes negative log density.

    SGHMC updates are used for non-zero momentum values. The gradient noise
    variance is assumed to be zero. Mass matrix is kept to be identity.

    WARN: The variance estimate of gradients is assumed to be zero for SGHMC.
    """

    def __init__(self, *args, momentum=0, temperature=1, **kwargs):
        super().__init__(*args, momentum=momentum, **kwargs)

        self.T = temperature
        # self.weight_decay = weight_decay
        if momentum != 0:
            self.resample_momentum()

    @torch.no_grad()
    def step(self, closure=None, noise=True):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            # weight_decay = self.weight_decay
            momentum = group['momentum']
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'].to('cuda'))
            sgld(params_with_grad,
                 d_p_list,
                 momentum_buffer_list,
                 weight_decay=weight_decay,
                 lr=lr,
                 momentum=momentum,
                 noise=noise,
                 temperature=self.T)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    @torch.no_grad()
    def resample_momentum(self):
        for group in self.param_groups:
            momentum = group['momentum']
            assert momentum > 0, "Must use momentum > 0 to use SGHMC."

            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.randn_like(p)
        return self
