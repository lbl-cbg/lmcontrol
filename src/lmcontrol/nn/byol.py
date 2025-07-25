import argparse
import copy
import glob
import os
import sys
import math

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*ToTensor().*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*EnumData.*")

import wandb

import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from lightning.pytorch.strategies import DDPStrategy

import numpy as np

from hdmf_ai import ResultsTable
from hdmf.common import get_hdf5io

import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from optuna.integration import PyTorchLightningPruningCallback

from ..utils import get_logger, parse_seed, format_time_diff
from .dataset import LMDataset, get_transforms as _get_transforms
from .resnet import ResNet, Bottleneck, get_block, get_planes, get_layers
from .utils import get_loaders


class MultiViewTransform:
    """Transforms an image into multiple views.

    Args:
        transforms:
            A sequence of transforms. Every transform creates a new view.

    """

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        """Transforms an image into multiple views.

        Every transform in self.transforms creates a new view.

        Args:
            image:
                Image to be transformed into multiple views.

        Returns:
            List of views.

        """
        images = list()
        for transform in self.transforms:
            tmp = transform(image, mask)
            images.append(tmp[0])
        return images
        #return [transform(image, mask) for transform in self.transforms]


class BYOL(L.LightningModule):

    # Note: The model and training settings do not follow the reference settings
    # from the paper. The settings are chosen such that the example can easily be
    # run on a small dataset with a single GPU.

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, total_steps=None, base_lr=0.01, min_lr=0.0001, planes=[8, 16, 32, 64], layers=[1, 1, 1, 1], block=Bottleneck):
        super().__init__()

        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

        self.base_momentum = 0.996
        self.final_momentum = 1.0
        self.backbone = ResNet(block=block, layers=layers, planes=planes, num_outputs=0, return_embeddings=True)
        self.projection_head = BYOLProjectionHead(self.backbone.n_features, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def cosine_lr_schedule(self, step, total_steps, start_value, end_value):
        """Cosine schedule for either learning rate or momentum."""
        return end_value + 0.5 * (start_value - end_value) * (
            1 + math.cos(math.pi * step / total_steps)
        )

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.global_step, self.total_steps, self.base_momentum, self.final_momentum)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.train_metric, loss, batch_size=x0.size(0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.val_metric, loss, batch_size=x0.size(0), sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        return self.backbone(x).flatten(start_dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.base_lr) #, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: self.cosine_lr_schedule(
                    step, self.total_steps, 1.0, self.min_lr/self.base_lr)
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': "step",
                "frequency": 1,       # Call scheduler every epoch
            }
        }


def _get_trainer(args, trial=None, extra_callbacks=None):
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
        filename="checkpoint-{epoch:02d}-{validation_ncs:.4f}",
        save_top_k=3,
        monitor="validation_ncs",
        mode="min",
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
        monitor="validation_ncs",
        patience=20,
        min_delta=0.001,
        mode="min"
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

    return L.Trainer(**targs)


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="HDMF input file")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--validation", type=str, nargs='+', help="directories containing validation data")
    grp.add_argument("--val_frac", type=float, default=None, help="Part of data to use for training (between 0 and 1)")

    parser.add_argument("--split-seed", type=parse_seed, help="seed for dataset splits", default='')

    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="disable parallel data loading", default=False)
    parser.add_argument("-q", "--quick", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-N", "--n_samples", type=int, help="number of samples to use from each NPZ", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)
    parser.add_argument("--wandb", action='store_true', default=False, help="provide this flag to stop wandb")
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("-b", "--batch-size", type=int, help="batch size for training and validation", default=32)
    parser.add_argument("--block", type=str, choices=['BasicBlock', 'Bottleneck'], help="type of block to use in the model", default='Bottleneck')
    parser.add_argument("--planes", type=str, choices=['3', '4'], help="list of number of planes for each layer", default='4')
    parser.add_argument("--layers", type=str, choices=['1', '2', '3', '4'], help="list of number of layers in each stage", default='4')
    parser.add_argument("--accelerator", type=str, help="type of accelerator for trainer", default="gpu")
    parser.add_argument("--strategy", type=str, help="type of strategy for trainer", default="auto")
    parser.add_argument("-g", "--devices", type=int, help="number of devices for trainer", default=1)
    parser.add_argument("-n", "--num_nodes", type=int, help="number of nodes for trainer", default=1)
    parser.add_argument("--exp-split", action='store_true', default=False, help="split samples using experimental conditions, otherwise randomly")

    args = parser.parse_args(argv)

    if args.num_nodes == 0:
        args.num_nodes = int(os.environ.get('SLURM_NNODES'))
        if args.num_nodes is None:
            print("Unable to determine number of nodes. Make sure you are running in a SLURM job environment when setting --num_nodes=0", file=sys.stderr)
            exit(1)

    args.block = get_block(args.block)
    args.planes = get_planes(args.planes)
    args.layers = get_planes(args.layers)

    logger = get_logger('info')

    logger.info(args)

    train_transform = MultiViewTransform(
        _get_transforms('float', 'norm', 'rotate', 'random_crop', 'hflip', 'vflip', 'rgb'),
        _get_transforms('float', 'norm', 'blur', 'rotate', 'random_crop', 'hflip', 'vflip', 'noise', 'rgb'),
    )

    val_transform = MultiViewTransform(
            _get_transforms('float', 'norm', 'rotate', 'random_crop', 'hflip', 'vflip', 'rgb'),
            _get_transforms('float', 'norm', 'random_crop', 'rgb'),
    )

    train_loader, val_loader = get_loaders(args,
                                           inference=False,
                                           train_tfm=train_transform,
                                           val_tfm=val_transform,
                                           exp_split=args.exp_split,
                                           return_labels=False,
                                           logger=logger)

    logger.info(f"{len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")

    dataset_size = len(train_loader.dataset)
    batches_per_epoch = math.ceil(dataset_size / args.batch_size)
    total_steps = args.epochs * batches_per_epoch

    model = BYOL(total_steps)

    timer = Timer()
    trainer = _get_trainer(args, extra_callbacks=[timer])
    logger.info(str(trainer))
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)

    total = timer.time_elapsed('train') + timer.time_elapsed('validate')
    logger.info(f"Took {total}")
    logger.info(f"  - training: {format_time_diff(timer.time_elapsed('train'))}")
    logger.info(f"  - validation: {format_time_diff(timer.time_elapsed('validate'))}")

def predict(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="HDMF input file")
    parser.add_argument("checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("output", type=str, help="the HDMF-AI file to save outputs to")
    parser.add_argument("--split-seed", type=parse_seed, help="seed for dataset splits", default=None)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output")
    parser.add_argument("-N", "--n-samples", type=int, help="number of samples to use from each class", default=None)
    parser.add_argument("-b", "--batch-size", type=int, help="batch size for training and validation", default=64)
    parser.add_argument("-g", "--devices", type=int, help="number of devices for trainer", default=1)
    parser.add_argument("-V", "--add-viz", action='store_true', help="Compute UMAP embedding for visualiztion", default=False)
    parser.add_argument("-2", "--two-dim", action='store_true', help="Compute 2D UMAP embedding (default is 3D embedding)", default=False)

    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = _get_transforms('float', 'norm', 'center_crop', 'rgb')


    loader = get_loaders(args,
                         inference=True,
                         tfm=transform,
                         return_labels=False,
                         logger=logger)

    dataset = loader.dataset


    model = BYOL.load_from_checkpoint(args.checkpoint)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=args.devices, accelerator=accelerator)

    logger.info("Running predictions witih Lightning")
    predictions = trainer.predict(model, loader)
    predictions = torch.cat(predictions).numpy()

    logger.info("Saving output")

    t = ResultsTable("byol_all",
                     description="Embeddings from a ResNet trained with BYOL",
                     n_samples=len(dataset))

    if dataset.split_mask is not None:
        t.add_tvt_split(dataset.split_mask.numpy(),
                        description=f"Generated with torch.randperm, with seed {args.split_seed}")

    t.add_embedding(predictions,
                    description="ResNet features")

    if dataset.table is None:
        dataset.open()

    if args.add_viz:
        if accelerator == "gpu":
            from cuml import UMAP
        else:
            from umap import UMAP
        D = 2 if args.two_dim else 3
        umap = UMAP(n_components=D,
                    min_dist=0.1, metric='euclidean')
        logger.info(f"Calculating {D}D embedding for visualization")
        emb = umap.fit_transform(predictions)
        t.add_viz_embedding(emb,
                            description=f"UMAP embedding computed using {str(umap.get_params())}")

    with get_hdf5io(args.output, mode='w') as io:
        io.write(t)


if __name__ == '__main__':
    train()
