import argparse
import glob
import pickle
import os

from functools import partial
import lightning as L
import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn
import torch
import glob
import numpy as np

import optuna
import lightning as L

import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.models.resnet import BasicBlock, Bottleneck 
from typing import Type, Union, List, Optional, Callable, Any
from optuna.integration import PyTorchLightningPruningCallback

from ..utils import get_logger
from .dataset import LMDataset, get_transforms as _get_transforms
from lmcontrol.nn.resnet import _resnet  

def resnet(*, weights=None, progress=True, block=None, layers=None, planes=None,num_classes=None) :
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    return _resnet(block=block, layers=layers, planes=planes, num_classes=num_classes, weights=weights, progress=progress)

def encode_labels(labels, return_classes=True):
    """This is a wrapper for sklearn.preprocessing.LabelEncoder"""
    enc = LabelEncoder().fit(labels)
    if return_classes:
        return enc.transform(labels), enc.classes_
    else:
        return enc.transform(labels)
    
import numpy as np

def load_npzs(npzs, logger, n=None, label_types=None):
    """Load data from NPZ files generated from lmcontrol crop command"""
    masks = []
    images = []
    paths = []
    metadata = dict()

    for npz_path in npzs:
        logger.debug(f"Reading {npz_path}")
        npz = np.load(npz_path)
        
        total_samples = len(npz['masks'])
        
        if n is not None and total_samples > n:
            indices = np.random.permutation(total_samples)[:n]
        else:
            indices = np.arange(total_samples)
        
        masks.append(npz['masks'][indices])
        images.append(npz['images'][indices])
        paths.append(npz['paths'][indices])

        md_keys = set(npz.keys()) - {'paths', 'masks', 'images'}
        logger.debug(f"Found the following keys in {npz_path}: {' '.join(sorted(md_keys))}")
        
        for k in sorted(md_keys):
            if npz[k].ndim == 0:
                metadata.setdefault(k, []).extend([str(npz[k])] * len(indices))
            else:
                metadata.setdefault(k, []).extend(np.array(npz[k])[indices])

    logger.debug("Merging masks")
    masks = np.concatenate(masks, axis=0)
    logger.debug("Merging images")
    images = np.concatenate(images, axis=0)
    logger.debug("Merging paths")
    paths = np.concatenate(paths, axis=0)
    
    metadata = {k: np.array(v) for k, v in metadata.items()}

    target_len = len(masks)
    for k in metadata.keys():
        if len(metadata[k]) != target_len:
            logger.critical(f"Metadata '{k}' not found in all NPZ files")
            raise ValueError(f"Metadata '{k}' length mismatch: expected {target_len}, got {len(metadata[k])}")

    return masks, images, paths, metadata


class LMDataset(Dataset):

    def __init__(self, npzs, use_masks=False, return_labels=False, logger=None, transform=None, label_types=None,n=None):
        """
        Args:
            npzs (array-like)       : A list or tuple of paths to NPZ files containing cropped images
        """
        if not isinstance(npzs, (list, tuple, np.ndarray, torch.Tensor)):
            raise ValueError(f"Got unexpected type ({type(npzs)}) for argument 'npzs'. Must be an array-like")
        elif len(npzs) == 0:
            raise ValueError("Got empty array-like for argument 'npzs'")
        logger = logger or get_logger('warning')
        masks, images, paths, metadata = load_npzs(npzs, logger,n,label_types)
        if use_masks:
            self.data = masks
        else:
            self.data = images
        self.data = torch.from_numpy(self.data)[:, None, :, :]
        self.paths = tuple(paths)
        self.transform = transform

        if not isinstance(label_types, (tuple, list)):
            label_types = [label_types]

        self.labels = None
        self.label_classes = None
        self.label_types = None
        if return_labels:
            tmp = list()
            self.label_classes = list()
            self.label_types = list()
            for k in metadata:
                if k not in label_types:
                    continue
                self.label_types.append(k)
                labels, classes = encode_labels(metadata[k])
                self.label_classes.append(classes)
                tmp.append(labels)
            self.labels = torch.from_numpy(np.stack(tmp, axis=1))

    def __getitem__(self, i):
        ret = self.data[i]
        if self.transform is not None:
            ret = self.transform(ret)
        labels = -1 if self.labels is None else self.labels[i]
        return ret, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def index_to_filename(dataset, i):
        return dataset.paths[i]



class LightningResNet(L.LightningModule):

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, num_classes, lr=0.01, step_size=2, gamma=0.1, planes=[8, 16, 32, 64], layers=[1, 1, 1, 1], block=BasicBlock):
        super().__init__()

        weights = None 
        progress = True

        self.backbone = resnet(weights=None, progress=True, block=block, layers=layers, planes=planes,num_classes=num_classes) 
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.block = block
        self.planes = planes
        self.layers = layers
        
    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels[:, 0])

        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels[:, 0].cpu().numpy(), preds.cpu().numpy())
        self.log('train_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log('train_loss', loss, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels[:, 0])

        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels[:, 0].cpu().numpy(), preds.cpu().numpy())
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log('val_loss', loss, batch_size=images.size(0))
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  #Read the pytorch documentation to understand how its used
        x = batch[0]
        return self.forward(x).flatten(start_dim=1)


def get_transform():
    transform = _get_transforms('float', 'norm','blur','rotate', 'crop','hflip', 'vflip', 'noise', 'rgb')
    return transform

def get_block(block_type):
    return {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck
    }[block_type]

def get_planes(plane_cmd):
    p = int(plane_cmd)  
    return [2**p, 2**(p+1), 2**(p+2), 2**(p+3)]

def get_layers(layers_cmd):
    l = int(layers_cmd)  
    layers = [1 if i < l else 0 for i in range(4)]  
    return layers

def _add_training_args(parser):
    parser.add_argument('labels', type=str, help="the label to train with") #Let us currently only work with single label. We will add other labels in future (Andrew)
    parser.add_argument("--training", type=str, nargs='+', required=True, help="directories containing training data")
    parser.add_argument("--validation", type=str, nargs='+', required=True, help="directories containing validation data")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-n", "--n_samples", type=int, help="number of samples to use from each NPZ", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)                                # Added extra (in the follwoing lines of parsing )
    parser.add_argument("--step_size", type=int, help="step size for learning rate scheduler", default=10)
    parser.add_argument("--gamma", type=float, help="gamma for learning rate scheduler", default=0.1)
    parser.add_argument("--batch_size", type=int, help="batch size for training and validation", default=32)
    parser.add_argument("--block", type=get_block, choices=['BasicBlock', 'Bottleneck'], help="type of block to use in the model", default='Bottleneck')
    parser.add_argument("--planes", type=get_planes, choices=['3', '4'], help="list of number of planes for each layer", default='4')
    parser.add_argument("--layers", type=get_layers, choices=['1', '2', '3', '4'], help="list of number of layers in each stage", default='4')


def _get_loaders_and_model(args,  logger=None):
    transform = get_transform()

    train_files = args.training
    val_files = args.validation

    n = args.n_samples

    if logger is None:
        logger = get_logger("critical")

    logger.info(f"Loading training data: {len(train_files)} files")
    train_dataset = LMDataset(train_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)

    for i in range(train_dataset.labels.shape[1]):
            logger.info(train_dataset.label_types[i] + " - " + str(torch.unique(train_dataset.labels[:, i])) + str(train_dataset.label_classes))


    logger.info(f"Loading validation data: {len(val_files)} files")
    val_dataset = LMDataset(val_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)
    for i in range(val_dataset.labels.shape[1]):
        logger.info(val_dataset.label_types[i] + " - " + str(torch.unique(val_dataset.labels[:, i])) + str(val_dataset.label_classes))

    num_workers = 0 if args.debug else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)

    model = LightningResNet(num_classes=len(train_dataset.label_classes[0]),lr=args.lr, step_size=args.step_size, gamma=args.gamma, block=args.block, planes=args.planes, layers=args.layers)

    return train_loader, val_loader, model


def _get_trainer(args, trial=None):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks = []

    targs = dict(max_epochs=args.epochs, devices=1, accelerator=accelerator, check_val_every_n_epoch=4, callbacks=callbacks)

    if args.early_stopping:
        early_stopping = EarlyStopping(
         monitor="val_accuracy",
         min_delta=0.001,
         patience=3,
         verbose=False,
         mode="max"
        )
        callbacks.append(early_stopping)

    if args.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint,  
            filename="checkpoint-{epoch:02d}-{val_accuracy:.2f}",  
            save_top_k=3, 
            monitor="val_accuracy", 
            mode="max"  
        )
        callbacks.append(checkpoint_callback)
        
    if trial is not None :   # if 'trial' is passed in, assume we are using Optuna to do HPO        
        targs['logger'] = CSVLogger(args.outdir, name=args.experiment)
        if args.pruning:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_accuracy"))
    else:
        wandb.init(project="SX_HTY_Run1")
        targs['logger'] = WandbLogger(project='your_project_name', log_model=True)

    return L.Trainer(**targs)


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="the experiment name")

    _add_training_args(parser)

    args = parser.parse_args(argv)

    logger = get_logger('info')

    train_loader, val_loader, model = _get_loaders_and_model(args, logger=logger)
    trainer = _get_trainer(args)
    trainer.fit(model, train_loader, val_loader)

def objective(args, trial):
    args.batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.step_size = trial.suggest_int('step_size', 5, 15, step=5)
    args.gamma = trial.suggest_float('gamma', 0.1, 0.5)


    block_type = trial.suggest_categorical('block_type', ['BasicBlock', 'Bottleneck'])
    args.block = get_block(block_type)

    p = trial.suggest_categorical('planes', ['3', '4'])  
    args.planes = get_planes(p)

    l = trial.suggest_categorical('layers', ['1', '2', '3', '4'])  
    args.layers = get_layers(l)

    args.outdir = os.path.join(args.working_dir, "logs")   ### outdir referrenced here 
    args.experiment = f"trial_{trial.number:04d}"

    train_loader, val_loader, model = _get_loaders_and_model(args)
    trainer = _get_trainer(args, trial=trial)
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_accuracy"].item()


def tune(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", type=str, help="the SQLite database to use")
    parser.add_argument("-t", "--n_trials", type=int, help="the number of trials to run", default=1)
    parser.add_argument("--pruning", action='store_true', help="Enable Optuna pruning", default=False)
    parser.add_argument("-r", "--restart", action='store_true', help="Restart study", default=False)

    _add_training_args(parser)

    args = parser.parse_args(argv)

    logger = get_logger('info')
    logger.info(f"Will run {args.n_trials} trials")
    pkl = os.path.join(args.working_dir, "args.pkl")
    db = os.path.join(args.working_dir, "study.db")
    if not os.path.exists(pkl) or args.restart:
        os.makedirs(args.working_dir, exist_ok=True) #adding this exist_ok check for the pre existence of file 
        with open(pkl, 'wb') as file:
            pickle.dump(args, file)
        if args.restart and os.path.exists(db):
            os.remove(db)                    
    else:
        with open(pkl, 'rb') as file:
            args = pickle.load(file)

    obj = partial(objective, args)

    study = optuna.create_study(storage=f"sqlite:///{db}", study_name="study", load_if_exists=True, direction="maximize")

    study.optimize(obj, n_trials=args.n_trials)


def predict(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, help="the label to predict with") #Let us currently only work with single label. We will add other labels in future  (Andrew)
    parser.add_argument("--prediction", type=str, nargs='+', required=True, help="directories containing prediction data")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-o","--output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")
    parser.add_argument("-n", "--data_size", type=int, help="number of samples to use from each class", default=None)


    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = get_transform()

    predict_files = args.prediction
    n = args.data_size


    logger.info(f"Loading predicting data: {len(predict_files)} files")
    predict_dataset = LMDataset(predict_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)
    for i in range(predict_dataset.labels.shape[1]):
        logger.info(predict_dataset.label_types[i] + " - " + str(torch.unique(predict_dataset.labels[:, i])) + str(predict_dataset.label_classes))

    true_labels = predict_dataset.labels[:, 0]  

    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=3)

    model = LightningResNet.load_from_checkpoint(args.checkpoint)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)

    logger.info("Running predictions")
    predictions = trainer.predict(model, predict_loader)
    predictions = torch.cat(predictions).numpy()

    pred_labels = np.argmax(predictions, axis=1)  

    if true_labels is not None: 
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        conf_matrix = confusion_matrix(true_labels, pred_labels)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

    out_data = dict(predictions=predictions, true_labels=true_labels)  

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    correct_incorrect = np.where(pred_labels == true_labels ,1 ,0) 
    out_data = dict(predictions = predictions, true_labels = true_labels, pred_labels = pred_labels, correct_incorrect = correct_incorrect)

    if not args.pred_only:
        dset = predict_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        for i, k in enumerate(dset.label_types):
            out_data[k + "_classes"] = dset.label_classes[i]
            out_data[k + "_labels"] = np.asarray(dset.labels[:, i])

  
    np.savez(args.output_npz, **out_data)

#misclassify function can be added if needed 

## merge this code with the optune branch ##

if __name__ == '__main__':
    train()