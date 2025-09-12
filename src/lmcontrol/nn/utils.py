import os
import sys

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, ModelSummary
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.strategies import DDPStrategy

import wandb

from .dataset import LMDataset
from ..utils import get_logger


def get_loaders(args, inference=True, tfm=None, train_tfm=None, val_tfm=None, return_labels=True, return_weights=False,
                logger=None):
    logger = logger or get_logger('warning')

    dset_kwargs = dict(logger=logger,
                       return_labels=return_labels,
                       exp_split=args.exp_split,
                       rand_split=not args.exp_split,
                       split_seed=args.split_seed)

    for attr in ('label', ):
        if hasattr(args, attr):
            dset_kwargs[attr] = getattr(args, attr)

    if args.debug:
        dl_kwargs = dict(batch_size=args.batch_size)
    else:
        dl_kwargs = dict(num_workers=5, batch_size=args.batch_size,
                         multiprocessing_context='spawn', persistent_workers=True)

    if inference:
        logger.info(f"Loading inference data from {args.input}")
        dataset = LMDataset(args.input, transform=tfm, **dset_kwargs)
        if args.debug:
            dataset.open()
            loader = DataLoader(dataset, **dl_kwargs)
        else:
            loader = DataLoader(dataset, worker_init_fn=dataset.open, **dl_kwargs)
        return loader
    else:
        logger.info(f"Loading training data from {args.input}")
        train_dataset = LMDataset(args.input, transform=train_tfm, split='train', return_weights=return_weights,
                                  **dset_kwargs)

        logger.info(f"Loading validation data from {args.input}")
        val_dataset = LMDataset(args.input, transform=val_tfm, split='validation', **dset_kwargs)

        if args.debug:
            train_dataset.open()
            val_dataset.open()
            train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
        else:
            train_loader = DataLoader(train_dataset, shuffle=True, worker_init_fn=train_dataset.open, **dl_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, worker_init_fn=val_dataset.open, **dl_kwargs)

        return train_loader, val_loader


def get_trainer(args, monitor, mode='min', trial=None, extra_callbacks=None):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks = []
    if extra_callbacks is not None and len(extra_callbacks) > 0:
        callbacks.extend(extra_callbacks)

    targs = dict(num_nodes=args.num_nodes, max_epochs=args.epochs, devices=args.devices,
                 accelerator="gpu" if args.devices > 0 else "cpu", check_val_every_n_epoch=2, callbacks=callbacks)

    if args.devices > 0:
        torch.set_float32_matmul_precision('medium')

    if args.devices > 1:  # If using multiple GPUs, use Distributed Data Parallel (DDP)
        targs['strategy'] = DDPStrategy(find_unused_parameters=True)

    # should we use r2 score and val_accuracy for measurement
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.outdir, 'best_ckpt'),
        filename="checkpoint-{epoch:02d}-{%s:.4f}" % monitor,
        save_top_k=3,
        monitor=monitor,
        mode=mode,
        save_last=True,
        every_n_epochs=1,
    )

    ckpt_steps = 10 if args.quick else 1000

    # Step-based checkpoint with step number in filename
    step_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.outdir, 'step_ckpt'),
        filename="step-checkpoint-{step}",
        save_top_k=1,                # Only keep 1 checkpoint
        every_n_train_steps=ckpt_steps,
        save_on_train_epoch_end=False,
        save_weights_only=False,     # Save the full model
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(step_checkpoint_callback)

    callbacks.append(LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True))

    early_stopping_callback = EarlyStopping(
        monitor=monitor,
        patience=20,
        min_delta=0.001,
        mode=mode
    )
    callbacks.append(early_stopping_callback)

    targs['logger'] = CSVLogger(args.outdir)
    if trial is not None :   # if 'trial' is passed in, assume we are using Optuna to do HPO
        if args.pruning:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="combined_metric"))
    else:
        if args.wandb:
            wandb.init(project="lmcontrol",
                       config=vars(args))
            targs['logger'] = WandbLogger(project='lmcontrol', log_model=True)

    if args.quick:
        targs['limit_train_batches'] = 100
        targs['limit_val_batches'] = 10

    for cb in callbacks:
        if isinstance(cb, ModelSummary):
            targs['enable_model_summary'] = True
            break

    return L.Trainer(**targs)
