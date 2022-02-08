"""SGLD pytorch Lightning module"""
import torch.nn as nn
import pytorch_lightning as pl
import torch
from .sghmc import SGLD as SGHMC
from .sgld import SGLD
from .lr_scheduler import CosineLR
import torchmetrics
from ..solo_learn.utils.lars import LARSWrapper


class SGLDModel(pl.LightningModule):
    def __init__(self, samples_dir: str, lr: float, epochs: int, temperature: float, momentum: float,
                 n_cycles: int, n_samples: int, weight_decay: float, N: int, criterion: torch.nn.Module,
                 is_sgld: bool, backbone: torch.nn.Module,
                 num_of_labels: int, num_of_batches: int, lars: bool = False, cyclic_lr: bool = False,
                 clip_val: float = 2, raw_params: bool = True,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.raw_params = raw_params
        self.cyclic_lr = cyclic_lr
        self.num_of_batches = num_of_batches
        self.lars = lars
        self.n_samples = n_samples
        self.n_cycles = n_cycles
        self.momentum = momentum
        self.temperature = temperature
        self.epochs = epochs
        self.samples_dir = samples_dir
        self.weight_decay = weight_decay
        self.N = N
        self.parameters_list = None
        self.is_sgld = is_sgld
        self.criterion = criterion
        self.backbone = backbone
        self.clip_val = clip_val
        self.num_of_labels = num_of_labels
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        elif hasattr(self.backbone, "num_features"):
            features_dim = self.backbone.num_features
        self.classifier = nn.Linear(features_dim, self.num_of_labels)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.automatic_optimization = False

    def write_metrices(self, prefix: str, metrices: dict, acc: float):
        self.log(f'{prefix}_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_loss", metrices['loss'], prog_bar=True)
        self.log(f"{prefix}_nll", metrices['nll'], prog_bar=True)
        self.log(f"{prefix}_prior", metrices['prior'], prog_bar=True)

    def training_step(self, batch, batch_idx):
        sgld_scheduler = self.lr_schedulers()
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x, y = batch
        y_hat = self.forward(x)
        if self.raw_params:
            params = torch.flatten(torch.cat([torch.flatten(p) for p in self.backbone.parameters()])).to('cuda')
        else:
            params = torch.flatten(torch.cat([torch.flatten(self.backbone.state_dict()[p]) for p in self.backbone.state_dict()])).to('cuda')

        metrices = self.criterion(y_hat, y, N=self.N, params=params)
        self.train_acc(y_hat, y)
        acc = self.train_acc
        self.write_metrices('train', metrices, acc)
        self.manual_backward(metrices['loss'])
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_val)

        if self.is_sgld:
            if self.cyclic_lr:
                if sgld_scheduler.get_last_beta() < sgld_scheduler.beta:
                    optimizer.step(noise=False)
                else:
                    optimizer.step()
            else:
                optimizer.step()
            if self.cyclic_lr and sgld_scheduler.should_sample():
                torch.save(self.state_dict(), self.samples_dir / f's_e{self.current_epoch}_m{batch_idx}.pt')
        else:
            optimizer.step()
        sgld_scheduler.step()
        torch.cuda.empty_cache()
        return metrices['loss']

    def forward(self, x):
        feats = self.backbone(x)
        y_hat = self.classifier(feats)
        return y_hat

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if self.raw_params:
            params = torch.flatten(torch.cat([torch.flatten(p) for p in self.backbone.parameters()])).to('cuda')
        else:
            params = torch.flatten(torch.cat([torch.flatten(self.backbone.state_dict()[p]) for p in self.backbone.state_dict()])).to('cuda')


        metrices = self.criterion(y_hat, y, N=self.N, params=params)
        acc = self.valid_acc(y_hat, y)
        self.write_metrices('val', metrices, acc)
        return {"loss": metrices['loss'], "acc": acc, 'nll': metrices['nll'],
                'prior': metrices['prior']}

    def validation_epoch_end(self, outputs):
        keys_val = outputs[0].keys()
        for key_val in keys_val:
            val = torch.stack([x[key_val] for x in outputs]).mean()
            self.log(f"val_{key_val}", val, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.is_sgld:

            optimizer = SGLD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                           temperature=self.temperature / self.N,
            )
            if self.cyclic_lr:
                scheduler = CosineLR(optimizer, n_cycles=self.n_cycles, n_samples=self.n_samples,
                                     T_max=self.num_of_batches * self.epochs)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_of_batches * self.epochs)
        else:
            optimizer = torch.optim.SGD
            optimizer = optimizer(
                self.parameters(),
                lr=self.lr,
                nesterov=True,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            if self.lars:
                optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_of_batches * self.epochs)
        return [optimizer], [scheduler]
