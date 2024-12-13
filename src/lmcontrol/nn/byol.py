import argparse
import copy
import glob
import os
import sys
import wandb

import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import BYOLTransform
from lightly.utils.scheduler import cosine_schedule
from optuna.integration import PyTorchLightningPruningCallback

from ..utils import get_logger
from .dataset import LMDataset, get_transforms as _get_transforms
from .resnet import ResNet, BasicBlock, Bottleneck
from .utils import get_loaders


class BYOL(L.LightningModule):

    # Note: The model and training settings do not follow the reference settings
    # from the paper. The settings are chosen such that the example can easily be
    # run on a small dataset with a single GPU.

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, lr=0.06, planes=[8, 16, 32, 64], layers=[1, 1, 1, 1], block=Bottleneck):
        super().__init__()

        self.backbone = ResNet(block=block, layers=layers, planes=planes, num_outputs=0, return_embeddings=True)
        self.projection_head = BYOLProjectionHead(self.backbone.n_features, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

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
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.train_metric, loss, batch_size=x0.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.val_metric, loss, batch_size=x0.size(0))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        return self.backbone(x).flatten(start_dim=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def _get_trainer(args, trial=None):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks = []

    targs = dict(num_nodes=args.num_nodes, max_epochs=args.epochs, devices=args.devices,
                 accelerator="gpu" if args.devices > 0 else "cpu", check_val_every_n_epoch=4, callbacks=callbacks)

    if args.devices > 1:  # If using multiple GPUs, use Distributed Data Parallel (DDP)
        targs['strategy'] = "ddp"

    # should we use r2 score and val_accuracy for measurement
    if args.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint,
            filename="checkpoint-{epoch:02d}-{validation_ncs:.4f}",
            save_top_k=3,
            monitor="validation_ncs",
            mode="min"
        )
        callbacks.append(checkpoint_callback)

    early_stopping_callback = EarlyStopping(
        monitor="validation_ncs",
        patience=5,
        min_delta=0.001,
        mode="min"
    )
    callbacks.append(early_stopping_callback)


    if trial is not None :   # if 'trial' is passed in, assume we are using Optuna to do HPO
        targs['logger'] = CSVLogger(args.outdir, name=args.experiment)
        if args.pruning:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="combined_metric"))

    else:
        if args.wandb:
            wandb.init(project="SX_HTY_Run1")
            targs['logger'] = WandbLogger(project='your_project_name', log_model=True)

    return L.Trainer(**targs)


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=str, nargs='+', required=True, help="directories containing training data")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--validation", type=str, nargs='+', help="directories containing validation data")
    grp.add_argument("--val_frac", type=float, default=None, help="Part of data to use for training (between 0 and 1)")

    parser.add_argument("--seed", type=int, help="seed for dataset splits")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-n", "--n_samples", type=int, help="number of samples to use from each NPZ", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)
    parser.add_argument("--wandb", action='store_false', default=True, help="provide this flag to stop wandb")
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--batch_size", type=int, help="batch size for training and validation", default=32)
    parser.add_argument("--block", type=get_block, choices=['BasicBlock', 'Bottleneck'], help="type of block to use in the model", default='Bottleneck')
    parser.add_argument("--planes", type=get_planes, choices=['3', '4'], help="list of number of planes for each layer", default='4')
    parser.add_argument("--layers", type=get_layers, choices=['1', '2', '3', '4'], help="list of number of layers in each stage", default='4')
    parser.add_argument("--accelerator", type=str, help="type of accelerator for trainer", default="gpu")
    parser.add_argument("--strategy", type=str, help="type of strategy for trainer", default="auto")
    parser.add_argument("--devices", type=int, help="number of devices for trainer", default=1)
    parser.add_argument("--num_nodes", type=int, help="number of nodes for trainer", default=4)

    args = parser.parse_args(argv)

    logger = get_logger('info')

    train_transform = BYOLTransform(
        view_1_transform=_get_transforms('float', 'norm', 'rotate', 'crop', 'hflip', 'vflip', 'rgb'),
        view_2_transform=_get_transforms('float', 'norm', 'blur', 'rotate', 'crop', 'hflip', 'vflip', 'noise', 'rgb'),
    )

    val_transform = BYOLTransform(
            view_1_transform=_get_transforms('float', 'norm', 'rotate', 'crop', 'hflip', 'vflip', 'rgb'),
            view_2_transform=_get_transforms('float', 'norm', 'crop', 'rgb'),
    )


    train_loader, val_loader = get_loaders(args,
                                           train_tfm=train_transform,
                                           val_tfm=val_transform,
                                           return_labels=False)

    model = BYOL()

    trainer = _get_trainer(args)
    trainer.fit(model, train_loader, val_loader)

def predict(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("prediction", type=str, nargs='+', required=True, help="directories containing prediction data")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")
    parser.add_argument("-n", "--n_samples", type=int, help="number of samples to use from each class", default=None)

    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = _get_transforms('float', 'norm', 'crop', 'rgb')

    logger.info(f"Loading inference data: {len(args.prediction)} files")
    test_dataset = LMDataset(args.prediction, transform=transform, logger=logger, return_labels=False, n_samples=args.n_samples)

    test_dl = DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=3)

    model = BYOL.load_from_checkpoint(args.checkpoint)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)

    logger.info("Running predictions witih Lightning")
    predictions = trainer.predict(model, test_dl)
    predictions = torch.cat(predictions).numpy()

    out_data = dict(predictions=predictions)

    if not args.pred_only:
        dset = test_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        out_data['metadata'] = {key: np.asarray(dset.metadata[key]) for key in dset.metadata}

    logger.info("Saving output")

    np.savez(args.output_npz, **out_data)


if __name__ == '__main__':
    train()
