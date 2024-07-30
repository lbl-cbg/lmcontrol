import argparse
import copy
import glob

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
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

from ..utils import get_logger
from .dataset import get_lightly_dataset, get_transforms as _get_transforms


class BYOL(L.LightningModule):

    # Note: The model and training settings do not follow the reference settings
    # from the paper. The settings are chosen such that the example can easily be
    # run on a small dataset with a single GPU.

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, model="resnet18"):
        super().__init__()
        if model == "resnet18":
            resnet = torchvision.models.resnet18()
            n_features = 512
        elif model == "resnet50":
            resnet = torchvision.models.resnet50()
            n_features = 2048
        elif model == "convnext_tiny":
            resnet = torchvision.models.convnext_tiny()
            n_features = 768
        else:
            raise ValueError(f"Unrecognized model: '{model}'")


        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(n_features, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()
        self.save_hyperparameters()

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
        (x0, x1) = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.train_metric, loss, batch_size=x0.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1) = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.val_metric, loss, batch_size=x0.size(0))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  #.predict function automatically call this
        x = batch[0]
        return self.backbone(x).flatten(start_dim=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)


def get_transform(transform1=None, transform2=None):
    """Get BYOL transform

    By default, the two transforms are:
        transform1: rotate, crop, hflip, vflip, float, rgb
        transform2: blur, rotate, crop, hflip, vflip, float, noise, rgb

    For more details on these transforms, see lmcontrol.nn.dataset.get_transform
    """
    if transform1 is None:
        transform1 = _get_transforms('float', 'norm', 'rotate', 'crop', 'hflip', 'vflip', 'rgb')
    if transform2 is None:
        transform2 = _get_transforms('float', 'norm', 'blur', 'rotate', 'crop', 'hflip', 'vflip', 'noise', 'rgb')

    transform = BYOLTransform(
        view_1_transform=transform1,
        view_2_transform=transform2
    )
    return transform


def get_npzs(timepoints, hts):
    ret = list()
    for tp in timepoints:
        for ht in hts:
            ret.extend(glob.glob(f"S{tp}/*HT{ht}/*.npz"))
    return ret


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint file to pick up from", default=None)
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)

    args = parser.parse_args(argv)

    logger = get_logger('info')


    if args.debug:
        train_files = get_npzs(["14", "4"], ["1"])
        val_files = get_npzs(["10"], ["5"])
        num_workers = 0
    else:
        train_files = get_npzs(["4", "14"], ["1", "2", "3", "4", "6", "7", "8", "9", "11", "12"])
        val_files = get_npzs(["10"], ["5", "10"])
        num_workers = 3

    train_tfm = get_transform()
    val_tfm = get_transform(
            transform1=_get_transforms('float', 'norm', 'rotate', 'crop', 'hflip', 'vflip', 'rgb'),
            transform2=_get_transforms('float', 'norm', 'crop', 'rgb'),
            )

    logger.info(f"Loading training data: {len(train_files)} files")
    train_dataset = get_lightly_dataset(train_files, transform=train_tfm, logger=logger)
    logger.info(f"Loading validation data: {len(val_files)} files")
    val_dataset = get_lightly_dataset(val_files, transform=val_tfm, logger=logger)

    model = BYOL()

    train_dl = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = L.Trainer(max_epochs=args.epochs, devices=1, accelerator=accelerator,
                        logger=CSVLogger(args.outdir, name=args.experiment),
                        callbacks=[EarlyStopping(monitor=model.val_metric, min_delta=0.001, patience=3, mode="min")])

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)



def predict(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False,
                        help="only save predictions, otherwise save original image data and labels in output_npz")

    args = parser.parse_args(argv)

    logger = get_logger('info')

    if args.debug:
        test_files = sorted(glob.glob(f"S4/*HT*1/*.npz"))
    else:
        test_files = sorted(glob.glob(f"S*/*HT*/*.npz"))

    transform = _get_transforms('float', 'norm', 'crop', 'rgb')
    logger.info(f"Loading training data: {len(test_files)} files")
    test_dataset = get_lightly_dataset(test_files, transform=transform, logger=logger, return_labels=True)

    test_dl = DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=3)

    model = BYOL.load_from_checkpoint(args.checkpoint)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)

    logger.info("Running predictions witih Lightning")
    predictions = trainer.predict(model, test_dl)
    predictions = torch.cat(predictions).numpy()

    out_data = dict(predictions=predictions)

    if not args.pred_only:
        dset = test_dataset.dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        for i, k in enumerate(dset.label_types):
            out_data[k + "_classes"] = dset.label_classes[i]
            out_data[k + "_labels"] = np.asarray(dset.labels[:, i])

    np.savez(args.output_npz, **out_data)


if __name__ == '__main__':
    train()