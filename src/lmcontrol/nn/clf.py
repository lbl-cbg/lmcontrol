import argparse
import glob
import pickle
import os
from typing import Type, Union, List, Optional, Callable, Any

from functools import partial
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import glob
import numpy as np

import optuna

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelSummary, Timer
from torchvision.models.resnet import BasicBlock, Bottleneck

from hdmf_ai.results_table import ResultsTable, RegressionOutput
from hdmf.common import get_hdf5io

from ..utils import get_logger, parse_seed, format_time_diff, get_metadata_info
from ..data_utils import load_npzs, encode_labels
from .dataset import LMDataset, get_transforms as _get_transforms
from .resnet import ResNet, add_args as add_resnet_args
from .utils import get_loaders, get_trainer


class MultiLabelLoss(nn.Module):

    def __init__(self, criteria, weights=None):
        super().__init__()
        self.criteria = criteria
        self.weights = torch.ones(len(criteria)) if weights is not None else torch.as_tensor(weights)

    def forward(self, input, target):
        ret = list()
        for _input, _target, _criterion, _weight in zip(input, target, self.criteria, self.weights):
            ret.append(_weight * _criterion(_input, _target))
        return ret


class CrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.nlll = nn.NLLLoss()

    def forward(self, input, target):
        return self.nlll(torch.log(input), target)


class MDNHead(nn.Module):

    def __init__(self, input_dim, n_components=2):
        super().__init__()
        self.n_components = n_components
        self.pi = nn.Linear(input_dim, n_components)        # mixing coefficients
        self.mu = nn.Linear(input_dim, n_components)        # means
        self.sigma = nn.Linear(input_dim, n_components)     # stddevs

    def forward(self, x):
        pi = F.softmax(self.pi(x), dim=-1)                  # mixing coefficients sum to 1
        mu = self.mu(x)                                     # means
        sigma = torch.exp(self.sigma(x))                    # positive stddevs
        return pi, mu, sigma



class MultivariateMDNHead(nn.Module):

    def __init__(self, input_dim, output_dim, n_components=2):
        super().__init__()
        self.output_dim = output_dim
        self.n_components = n_components

        self._n_chol = (output_dim * (output_dim + 1) // 2)
        self._L_diag_idx = torch.roll(torch.cumsum(torch.arange(output_dim), dim=0) - 1, -1)

        self.pi = nn.Linear(input_dim, n_components)                            # mixing coefficients
        self.mu = nn.Linear(input_dim, n_components * output_dim)               # means
        self.sigma_L = nn.Linear(input_dim, n_components * self._n_chol)        # Cholesky factors

    def forward(self, x):
        pi = F.softmax(self.pi(x), dim=-1)                                      # mixing coefficients sum to 1
        mu = self.mu(x).reshape(-1, self.n_components, self.output_dim)         # means

        L = self.sigma_L(x).reshape(-1, self.n_components, self._n_chol)        # Cholesky factors
        L[:, :, self._L_diag_idx] = torch.exp(L[:, :, self._L_diag_idx])        # positive diagonal factors

        return pi, mu, L


time_weight = 0.9

class LightningResNet(L.LightningModule):


    loss_functions = {
        'time': nn.MSELoss(),
        'sample': CrossEntropy(),
        'condition': CrossEntropy(),
        'feed': CrossEntropy(),
        'starting_media': CrossEntropy()
    }
    loss_weights = {
        'time': time_weight,
        'sample': 0,
        'condition': 0,
        'feed': 1-time_weight,
        'starting_media': 0
    }


    def __init__(self, label_classes, lr=0.01, step_size=2, gamma=0.1, planes=[8, 16, 32, 64],
                 layers=[1, 1, 1, 1], block=BasicBlock, include_features=False, features_only=False, time_weight=1e-3):
        super().__init__()

        self.label_classes = label_classes

        self.label_counts = [1 if x is None else len(x) for x in self.label_classes.values()]

        self.num_outputs = sum(self.label_counts)

        self.weighted_multilabel = False

        if self.num_outputs == len(self.label_classes):
            # Assume mult-label with the same loss type
            self.criterion = nn.MSELoss()
            self.activations = None
        else:
            self.weighted_multilabel = True
            # Keep this here, since it's for multilabel loss when we have different loss types
            self.criterion = MultiLabelLoss([self.loss_functions.get(l, nn.MSELoss()) for l in self.label_classes],
                                            weights=[self.loss_weights.get(l, 1.0) for l in self.label_classes])
            self.activations = list()
            for l in self.label_classes:
                if LMDataset.is_regression(l):
                    if l == 'time':
                        self.activations.append(nn.Sequential(nn.Softplus(), nn.Flatten(start_dim=0)))
                    else:
                        self.activations.append(nn.Flatten(start_dim=0))
                else:
                    self.activations.append(nn.Softmax(dim=1))

        self.backbone = ResNet(block=block, layers=layers, planes=planes, num_outputs=self.num_outputs,
                               include_features=include_features, features_only=features_only)

        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.block = block
        self.planes = planes
        self.layers = layers
        self.features_only = features_only
        self.save_hyperparameters()

    def forward(self, x):
        outputs = self.backbone(x)
        if self.weighted_multilabel:
            # If we are doing multi-label
            ret = list()
            start_idx = 0
            for act, count in zip(self.activations, self.label_counts):
                end_idx = start_idx + count
                ret.append(act(outputs[:, start_idx:end_idx]))
                start_idx = end_idx
            return tuple(ret)
        else:
            return outputs


    def _score(self, step_type, outputs, loss_components, true_labels):
        total_r2 = 0
        total_accuracy = 0
        regression_tasks = 0
        classification_tasks = 0


        for _label_type, _output, _loss, _label in zip(self.label_classes, outputs, loss_components, true_labels):
            _classes = self.label_classes[_label_type]
            self.log(f"{step_type}_{_label_type}_loss", _loss, on_step=False, on_epoch=True)

            if _classes is None:
                r2 = r2_score(_label.cpu().detach().numpy(), _output.cpu().detach().numpy())
                self.log(f"{step_type}_{_label_type}_r2", r2, on_step=False, on_epoch=True)
                total_r2 += r2
                regression_tasks += 1
            else:
                preds = torch.argmax(_output, dim=1)
                acc = accuracy_score(_label.cpu().numpy(), preds.cpu().numpy())
                self.log(f"{step_type}_{_label_type}_accuracy", acc, on_step=False, on_epoch=True)
                total_accuracy += acc
                classification_tasks += 1

        if regression_tasks > 0:
            mean_r2 = total_r2 / regression_tasks
            self.log(f'{step_type}_mean_r2', mean_r2, on_step=False, on_epoch=True)
        if classification_tasks > 0:
            mean_accuracy = total_accuracy / classification_tasks
            self.log(f'{step_type}_mean_accuracy', mean_accuracy, on_step=False, on_epoch=True)



    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)

        total_loss = self.criterion(outputs, labels)

        if isinstance(self.criterion, MultiLabelLoss):
            # Keep this here, since it's for multilabel loss when we have different loss types
            loss_components = self.criterion(outputs, labels)
            self._score("train", outputs, loss_components, labels)
            total_loss = sum(loss_components)
            self.log('total_train_loss', total_loss)
        else:
            self.log('train_loss', total_loss)


        return total_loss


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)

        total_loss = self.criterion(outputs, labels)

        if isinstance(self.criterion, MultiLabelLoss):
            # Keep this here, since it's for multilabel loss when we have different loss types
            loss_components = self.criterion(outputs, labels)
            self._score("val", outputs, loss_components, labels)
            total_loss = sum(loss_components)
            self.log('total_val_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        else:
            self.log('val_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=self.gamma)
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        return self.forward(x)


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
    parser.add_argument("input", help="HDMF input file")
    parser.add_argument('label', type=str, nargs='+', choices=['fcs', 'time', 'feed', 'starting_media', 'condition', 'sample'], help="the label to train with")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--validation", type=str, nargs='+', help="directories containing validation data")
    grp.add_argument("--val_frac", type=float, default=None, help="Part of data to use for training (between 0 and 1)")

    parser.add_argument("--split-seed", type=parse_seed, help="seed for dataset splits", default=None)
    parser.add_argument("--exp-split", action='store_true', default=False, help="split samples using experimental conditions, otherwise randomly")

    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)
    parser.add_argument("--wandb", action='store_false', default=False, help="provide this flag to stop wandb")
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--step_size", type=int, help="step size for learning rate scheduler", default=10)
    parser.add_argument("--gamma", type=float, help="gamma for learning rate scheduler", default=0.1)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size for training and validation", default=32)
    parser.add_argument("--time_weight", type=float, help="loss function weight for time", default=0.001)
    parser.add_argument("-g", "--devices", type=int, help="number of devices for trainer", default=1)
    parser.add_argument("-n", "--num_nodes", type=int, help="number of nodes for trainer", default=1)

    parser.add_argument("-q", "--quick", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-d", "--debug", action='store_true', help="dismantled parallel data loading", default=False)

    add_resnet_args(parser)


def _get_loaders_and_model(args,  logger=None):
    train_transform = _get_transforms('float', 'norm', 'blur','rotate', 'random_crop','hflip', 'vflip', 'noise', 'rgb')
    val_transform = _get_transforms('float', 'norm', 'center_crop', 'rgb')
    # val_transform = _get_transforms('float', 'norm','blur','rotate', 'random_crop','hflip', 'vflip', 'noise', 'rgb')

    train_loader, val_loader = get_loaders(args,
                                           inference=False,
                                           train_tfm=train_transform,
                                           val_tfm=val_transform,
                                           return_labels=True)


    model = LightningResNet(train_loader.dataset.label_classes, lr=args.lr, step_size=args.step_size, gamma=args.gamma,
                            block=args.block, planes=args.planes, layers=args.layers, time_weight=args.time_weight)

    return train_loader, val_loader, model


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", "--experiment", type=str, help="the experiment name")

    _add_training_args(parser)

    args = parser.parse_args(argv)

    logger = get_logger('info')

    train_loader, val_loader, model = _get_loaders_and_model(args, logger=logger)

    timer = Timer()
    mod_sum = ModelSummary(max_depth=3)
    trainer = get_trainer(args, 'val_loss' if args.label == ['fcs'] else 'total_val_loss', extra_callbacks=[timer, mod_sum])
    trainer.fit(model, train_loader, val_loader)

    total = timer.time_elapsed('train') + timer.time_elapsed('validate')
    logger.info(f"Took {total}")
    logger.info(f"  - training: {format_time_diff(timer.time_elapsed('train'))}")
    logger.info(f"  - validation: {format_time_diff(timer.time_elapsed('validate'))}")

def objective(args, trial):
    args.batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.step_size = trial.suggest_int('step_size', 5, 15, step=5)
    args.gamma = trial.suggest_float('gamma', 0.1, 0.5)
    args.time_weight = trial.suggest_float('time_weight', 1e-5, 1.0, log=True)

    block_type = trial.suggest_categorical('block_type', ['BasicBlock', 'Bottleneck'])
    args.block = get_block(block_type)

    p = trial.suggest_categorical('planes', ['3', '4'])
    args.planes = get_planes(p)

    l = trial.suggest_categorical('layers', ['1', '2', '3', '4'])
    args.layers = get_layers(l)

    args.outdir = os.path.join(args.working_dir, "logs")
    args.experiment = f"trial_{trial.number:04d}"

    train_loader, val_loader, model = _get_loaders_and_model(args)
    trainer = get_trainer(args, 'total_val_loss', trial=trial)
    trainer.fit(model, train_loader, val_loader)

    val_accuracy = trainer.callback_metrics.get("val_mean_accuracy", None)
    val_r2 = trainer.callback_metrics.get("val_mean_r2", None)

    return val_accuracy, val_r2

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
        os.makedirs(args.working_dir, exist_ok=True)
        with open(pkl, 'wb') as file:
            pickle.dump(args, file)
        if args.restart and os.path.exists(db):
            os.remove(db)
    else:
        with open(pkl, 'rb') as file:
            args = pickle.load(file)

    obj = partial(objective, args)

    study = optuna.create_study(storage=f"sqlite:///{db}", study_name="study", load_if_exists=True, directions=["maximize", "maximize"])

    study.optimize(obj, n_trials=args.n_trials)
    logger.info(f"Best trials: {study.best_trials}")

def predict(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("input", type=str, help="HDMF input file")
    parser.add_argument("output", type=str, help="the path to save HDMF-AI table to")
    parser.add_argument('label', type=str, nargs='+', choices=['fcs', 'time', 'feed', 'starting_media', 'condition', 'sample'], help="the label to predict with")
    parser.add_argument("-c", "--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("--split-seed", type=parse_seed, help="seed for dataset splits", default=None)
    parser.add_argument("--exp-split", action='store_true', default=False, help="split samples using experimental conditions, otherwise randomly")

    parser.add_argument("-b", "--batch_size", type=int, help="batch size for training and validation", default=32)

    parser.add_argument("--save_residuals", action='store_true', default=False, help="provide this if you want to store the residual values")
    parser.add_argument("-V", "--add-viz", action='store_true', help="Compute UMAP embedding for visualiztion", default=False)
    parser.add_argument("-2", "--two-dim", action='store_true', help="Compute 2D UMAP embedding (default is 3D embedding)", default=False)

    parser.add_argument("-q", "--quick", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-d", "--debug", action='store_true', help="dismantled parallel data loading", default=False)

    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = _get_transforms('float', 'norm', 'center_crop', 'rgb')

    metadata_info = get_metadata_info()
    logger.info(f"Loading prediction data: {len(args.input)} files")

    loader = get_loaders(args,
                         inference=True,
                         tfm=transform,
                         return_labels=True,
                         logger=logger)

    dataset = loader.dataset

    model = LightningResNet.load_from_checkpoint(args.checkpoint, label_classes=dataset.label_classes,
                                                 include_features=True, features_only=False)

    targs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "limit_predict_batches": 100 if args.quick else None,
        "devices": 1
    }
    trainer = L.Trainer(**targs)

    logger.info("Running predictions")

    predictions = trainer.predict(model, loader)
    features, predictions = [torch.cat(x).numpy() for x in zip(*predictions)]

    logger.info(f"Results saved to {args.output}")

    t = ResultsTable("resnet_supervised",
                     description="Results from supervised training a ResNet",
                     n_samples=len(dataset))

    if dataset.split_mask is not None:
        t.add_tvt_split(dataset.split_mask.numpy()[:len(features)],
                        description=f"Generated with torch.randperm, with seed {args.split_seed}")

    t.add_embedding(features, description="ResNet features")
    for i, c in enumerate(dataset.FC_COLS):
        t.add_column(c, description=metadata_info[c]['description'], data=predictions[:, i], col_cls=RegressionOutput)

    if dataset.table is None:
        dataset.open()

    if args.add_viz:
        if targs['accelerator'] == "gpu":
            from cuml import UMAP
        else:
            from umap import UMAP
        D = 2 if args.two_dim else 3
        umap = UMAP(n_components=D,
                    min_dist=0.1, metric='euclidean')
        logger.info(f"Calculating {D}D embedding for visualization")
        emb = umap.fit_transform(features)
        t.add_viz_embedding(emb,
                            description=f"UMAP embedding computed using {str(umap.get_params())}")

    with get_hdf5io(args.output, mode='w') as io:
        io.write(t)



if __name__ == '__main__':
    predict()
