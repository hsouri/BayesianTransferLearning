import os
from pathlib import Path
import datetime
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def load_loggers_callbacks(ignore_wandb: bool, use_mlflow: bool, use_tune: bool, config: dict, model):
    """Load the loggers and callbacks for ray tune, mlflow amd wandb"""
    loggers = []
    callbacks = []
    if not ignore_wandb:
        suffix = datetime.datetime.now().strftime("%y%m%d")
        save_path = os.path.join(config['wandb_save_dir'], suffix, config['wandb_project'], config['wandb_name'])
        Path(save_path).mkdir(parents=True, exist_ok=True)
        wandb_logger = WandbLogger(
            name=config['wandb_name'], project=config['wandb_project'], entity=config['wandb_entity'], 
            offline=False, save_dir=save_path, log_model='all')
        wandb_logger.log_hyperparams(config)
        # wandb_logger.watch(model, log="gradients", log_freq=100)

        loggers.append(wandb_logger)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        if config['save_checkpoints']:
            save_path = os.path.join(save_path, 'checkpoints')
            Path(save_path).mkdir(parents=True, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', dirpath=save_path)
            callbacks.append(checkpoint_callback)

    if use_mlflow:
        mlflow_logger = MLFlowLogger(experiment_name=str(config['experiment_name']), run_name=str(config['run_name']))
        mlflow_logger.log_hyperparams(config)
        loggers.append(mlflow_logger)
    if use_tune:
        tune_report = TuneReportCallback(
            {
                "loss": "ptl/val_loss",
                "mean_accuracy": "ptl/val_acc"
            },
            on="validation_end")
        callbacks.append(tune_report)
    return loggers, callbacks
