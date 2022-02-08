r"""
SWAG Callback
based on PYTORCH_LIGHTNING.CALLBACKS.STOCHASTIC_WEIGHT_AVG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import os
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Callable, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.optimizers import _get_default_scheduler_config
from torch.optim.swa_utils import SWALR
from ..sghmc.lr_scheduler import CosineLR
from .swag import SWAG
import math

_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


def get_num_training_steps(trainer) -> int:
    """Total training steps inferred from datamodule and devices."""
    if trainer.num_training_batches != float('inf'):
        dataset_size = trainer.num_training_batches
    else:
        dataset_size = len(trainer._data_connector._train_dataloader_source.dataloader())

    if isinstance(trainer.limit_train_batches, int):
        dataset_size = min(dataset_size, trainer.limit_train_batches)
    else:
        dataset_size = int(dataset_size * trainer.limit_train_batches)

    # accelerator_connector = trainer._accelerator_connector
    # if accelerator_connector.use_ddp2 or accelerator_connector.use_dp:
    #    effective_devices = 1
    # else:
    effective_devices = 1

    effective_devices = effective_devices * trainer.num_nodes
    effective_batch_size = trainer.accumulate_grad_batches * effective_devices
    effective_batch_size = 1
    max_estimated_steps = math.ceil(dataset_size // effective_batch_size) * trainer.max_epochs
    if trainer.max_steps != None:
        max_estimated_steps = min(max_estimated_steps,
                                  trainer.max_steps) if trainer.max_steps != -1 else max_estimated_steps
    return max_estimated_steps


def cross_entropy(model, input, target):
    # standard cross-entropy loss function
    output = model(input)
    logits = output["logits"]
    loss = F.cross_entropy(logits, target, ignore_index=-1)
    return loss, logits, {}


def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Adds basic momentum arguments that are shared for all methods.

    Args:
        parent_parser (ArgumentParser): argument parser that is used to create a
            argument group.

    Returns:
        ArgumentParser: same as the argument, used to avoid errors.
        :param self:
    """

    parser = parent_parser.add_argument_group("swas")
    # momentum settings
    parser.add_argument("--annealing_epochs", default=1, type=int)
    parser.add_argument("--use_swag", action='store_true')
    parser.add_argument("--swa_lrs", default=0.02, type=float)
    parser.add_argument("--start_swag", default=1, type=int)
    parser.add_argument("--interval_steps", default=1, type=int)
    parser.add_argument("--n_cycles", default=5, type=int)
    parser.add_argument("--n_samples", default=5, type=int)
    parser.add_argument("--swag_scheduler", default='cyclic', type=str)
    return parent_parser


class StochasticWeightAveraging(Callback):
    def __init__(
            self,
            swa_step_start: Union[int, float] = 0.8,
            swa_lrs: Optional[Union[float, List[float]]] = None,
            annealing_epochs: int = 10,
            annealing_strategy: str = "cos",
            avg_fn: Optional[_AVG_FN] = None,
            device: Optional[Union[torch.device, str]] = torch.device("cpu"),
            source_path: str = './log',
            interval_steps: int = 20,
            num_classes: int = 10,
            swag: bool = True,
            subspace: str = 'covariance',
            max_num_models: int = 20,
            criterion=cross_entropy,
            swag_model_name: str = 'swag_model1.pt',
            scheduler: str = 'cyclic',
            n_cycles: int = 3,
            n_samples: int = 3

    ):
        self.scheduler = scheduler
        self.swa_step_start = swa_step_start
        self._swa_lrs = swa_lrs
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn or avg_fn2
        self._device = device
        self._average_model = None
        self.swag = swag
        self.source_path = source_path
        self.interval_steps = interval_steps
        self.num_classes = num_classes
        self.subspace = subspace
        self.max_num_models = max_num_models
        self.criterion = criterion
        self.swag_model_name = swag_model_name
        self.n_cycles = n_cycles
        self.n_samples = n_samples

    @property
    def swa_start(self) -> int:
        return max(self.swa_step_start - 1, 0)  # 0-based

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.current_step = 0

    def on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # copy the model before moving it to accelerator device.
        with pl_module._prevent_trainer_and_dataloaders_deepcopy():
            if self.swag:
                self._average_model = SWAG(deepcopy(pl_module.encoder),
                                           subspace_type=self.subspace,
                                           subspace_kwargs={'max_rank': self.max_num_models},
                                           num_classes=self.num_classes)
                self._average_model.to(self._device or pl_module.device)
            else:
                self._average_model = deepcopy(pl_module)

    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int,
            dataloader_idx: int
    ) -> None:

        if self.current_step == self.swa_start:
            optimizer = trainer.optimizers[0]
            if self._swa_lrs is None:
                self._swa_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
            if isinstance(self._swa_lrs, float):
                self._swa_lrs = [self._swa_lrs] * len(optimizer.param_groups)
            for lr, group in zip(self._swa_lrs, optimizer.param_groups):
                group["initial_lr"] = lr
            if self.scheduler == 'constant':
                self._swa_scheduler = SWALR(
                    optimizer,
                    swa_lr=self._swa_lrs,
                    anneal_epochs=self._annealing_epochs,
                    anneal_strategy=self._annealing_strategy,
                    last_epoch=trainer.max_epochs if self._annealing_strategy == "cos" else -1,
                )
            else:
                num_training_steps = get_num_training_steps(trainer)
                self._swa_scheduler = CosineLR(optimizer, n_cycles=self.n_cycles, n_samples=self.n_samples,
                                               T_max=num_training_steps, last_epoch=trainer.max_epochs)
            default_scheduler_cfg = _get_default_scheduler_config()
            default_scheduler_cfg["scheduler"] = self._swa_scheduler
            if trainer.lr_schedulers:
                trainer.lr_schedulers[0] = default_scheduler_cfg
            else:
                trainer.lr_schedulers.append(default_scheduler_cfg)
            self.n_averaged = torch.tensor(0, dtype=torch.long, device=pl_module.device)

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if self.swa_start <= self.current_step:
            if self.swag:
                should_sample = False
                if self.scheduler == 'constant':
                    if self.current_step % self.interval_steps == 0:
                        should_sample = True
                else:
                    if self._swa_scheduler.should_sample():
                        should_sample = True
                if should_sample:
                    self._average_model.collect_model(pl_module.encoder)
                    # mean, variance = self._average_model._get_mean_and_variance()
                    file = os.path.join(self.source_path, self.swag_model_name)
                    torch.save(self._average_model, file)

            else:
                update_parameters(self._average_model, pl_module, self.n_averaged, self.avg_fn)
            self._swa_scheduler.step()
        self.current_step += 1



def update_parameters(
        average_model: "pl.LightningModule", model: "pl.LightningModule", n_averaged: torch.LongTensor, avg_fn: _AVG_FN
):
    """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L104-L112."""
    for p_swa, p_model in zip(average_model.parameters(), model.parameters()):
        device = p_swa.device
        p_swa_ = p_swa.detach()
        p_model_ = p_model.detach().to(device)
        src = p_model_ if n_averaged == 0 else avg_fn(p_swa_, p_model_, n_averaged.to(device))
        p_swa_.copy_(src)
    n_averaged += 1


def avg_fn2(
        averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: torch.LongTensor
) -> torch.FloatTensor:
    """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97."""
    return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)
