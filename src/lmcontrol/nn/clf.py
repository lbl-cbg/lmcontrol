import argparse
import glob
from itertools import product
import math
import pickle
import os
from typing import Type, Union, List, Optional, Callable, Any



from functools import partial
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import glob
import numpy as np
from sklearn.mixture import GaussianMixture

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


def compute_cov(sigma_L):
    D = int((math.sqrt(1 + 8 * sigma_L.shape[1]) - 1) // 2)
    L_sq = torch.zeros(sigma_L.shape[0], D, D, device=sigma_L.device, dtype=sigma_L.dtype)
    tril_idx = torch.tril_indices(D, D)
    L_sq[:, tril_idx[0], tril_idx[1]] = sigma_L
    return L_sq.matmul(L_sq.transpose(1, 2))



def init_mdn(data, random_state=42):
    """
    Find univariate modes for each variable using GMM, then return all combinations
    along with Cholesky factors for each covariance matrix

    Parameters:
    -----------
    data : numpy.ndarray
        Nx2 matrix where each column is a variable
    random_state : int
        Random state for reproducible results

    Returns:
    --------
    modes : numpy.ndarray
        4x2 matrix containing all combinations of univariate modes
        Each row is [mode_var1, mode_var2]
    chol_factors : numpy.ndarray
        4x3 array containing Cholesky factors of the covariances for combination of univariate modes
    """

    if data.shape[1] != 2:
        raise ValueError("Input data must be Nx2 matrix")

    # Fit GMM with 2 components for each variable
    gmm1 = GaussianMixture(n_components=2, random_state=random_state).fit(data[:, 0].reshape(-1, 1))
    gmm2 = GaussianMixture(n_components=2, random_state=random_state).fit(data[:, 1].reshape(-1, 1))

    # Extract modes (means of the components) and covariances
    modes_var1 = gmm1.means_.flatten()
    modes_var2 = gmm2.means_.flatten()

    # For univariate case, covariances are just variances (scalars)
    covariances_var1 = gmm1.covariances_.flatten()  # Shape: (2,)
    covariances_var2 = gmm2.covariances_.flatten()  # Shape: (2,)

    # Sort modes and covariances consistently
    sort_idx1 = np.argsort(modes_var1)
    sort_idx2 = np.argsort(modes_var2)

    modes_var1 = modes_var1[sort_idx1]
    modes_var2 = modes_var2[sort_idx2]

    covariances_var1 = covariances_var1[sort_idx1]
    covariances_var2 = covariances_var2[sort_idx2]

    # Generate all combinations of modes
    mode_combinations = list(product(modes_var1, modes_var2))
    modes_matrix = np.array(mode_combinations)

    # Generate all combinations of covariances
    # Create 2x2 covariance matrices assuming independence between variables
    chol_factors = []
    for cov1, cov2 in product(covariances_var1, covariances_var2):
        # Create diagonal covariance matrix (assuming independence)
        cov_matrix = np.array([[cov1, 0],
                              [0, cov2]])
        chol_factors.append(np.sqrt([cov1, 0, cov2]))

    chol_factors = np.array(chol_factors)

    return torch.tensor(modes_matrix), torch.tensor(chol_factors)


class GatedMultivariateMDNHead(nn.Module):

    def __init__(self, output_dim, n_components=4, init_mu=None, init_L=None):
        super().__init__()
        self.output_dim = output_dim
        self.n_components = n_components

        self._n_chol = (output_dim * (output_dim + 1) // 2)

        # Pre-compute diagonal indices
        L_diag_idx = (torch.arange(output_dim) + 1) * (torch.arange(output_dim) + 2) // 2 - 1
        self.register_buffer('_L_diag_idx', L_diag_idx)

        # Fixed component parameters (raw, unconstrained)
        mu_tmp = torch.randn(n_components, output_dim)
        if init_mu is not None:
            if init_mu.shape != mu_tmp.shape:
                raise ValueError(f"init_mu has incorrect shape: Got {init_mu.shape}, expected {mu_tmp.shape}")
            mu_tmp = init_mu.detach().clone().to(torch.float32)
        self.mu = nn.Parameter(mu_tmp)
        L_tmp = torch.randn(n_components, self._n_chol)
        if init_L is not None:
            if init_L.shape != L_tmp.shape:
                raise ValueError(f"init_L has incorrect shape: Got {init_L.shape}, expected {L_tmp.shape}")
            L_tmp = init_L.detach().clone().to(torch.float32)
            L_tmp[:, self._L_diag_idx] = torch.log(L_tmp[:, self._L_diag_idx])
        self._sigma_L_raw = nn.Parameter(L_tmp)

    @property
    def sigma_L(self):
        """Get constrained Cholesky factors"""
        L = self._sigma_L_raw.clone()
        L[:, self._L_diag_idx] = torch.exp(L[:, self._L_diag_idx])
        return L

    def forward(self, x):
        # Input-dependent mixing coefficients
        pi = F.softmax(x, dim=-1)

        return pi


class GatedMDNLoss(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.D = output_dim
        self.norm_const = -0.5 * self.D * math.log(2 * math.pi)

        # Pre-compute indices
        self.register_buffer('tril_idx', torch.tril_indices(output_dim, output_dim))
        # Pre-compute indices

    def forward(self, pi, mu, L_constrained, target):
        """
        pi             (batch_size, n_components)
        mu             (n_components, output_dims)
        L_constrained  (n_components, n_chol) - ALREADY CONSTRAINED
        target         (batch_size, output_dims)
        """
        batch_size, n_components = pi.shape

        # Build all Cholesky matrices at once (no constraints needed here!)
        L_sq = torch.zeros(n_components, self.D, self.D, device=target.device, dtype=target.dtype)
        L_sq[:, self.tril_idx[0], self.tril_idx[1]] = L_constrained

        # Compute differences for all components
        diff = target.unsqueeze(1) - mu.unsqueeze(0)  # (batch_size, n_components, D)

        # Solve triangular systems for all components
        z = torch.linalg.solve_triangular(
            L_sq.unsqueeze(0).expand(batch_size, -1, -1, -1),
            diff.unsqueeze(-1),
            upper=False
        ).squeeze(-1)  # (batch_size, n_components, D)

        # Mahalanobis distances
        M = torch.sum(z**2, dim=-1)  # (batch_size, n_components)

        # Log determinants for all components
        log_dets = torch.sum(torch.log(torch.diagonal(L_sq, dim1=-2, dim2=-1)), dim=-1)  # (n_components,)

        # Log probabilities
        log_probs = self.norm_const - log_dets.unsqueeze(0) - 0.5 * M

        # Weighted log probabilities
        weighted_log_probs = log_probs + torch.log(pi + 1e-10)

        # Negative log likelihood
        nll = -torch.mean(torch.logsumexp(weighted_log_probs, dim=1))

        return nll


class LightningResNet(pl.LightningModule):

    time_weight = 0.9

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


    def __init__(self, label_classes, lr=0.01, step_size=2, gamma=0.1, planes=[8, 16, 32, 64], layers=[1, 1, 1, 1],
                 block=BasicBlock, include_features=True, features_only=False, n_components=None, init_mu=None, init_L=None):
        super().__init__()

        self.label_classes = label_classes

        self.label_counts = [1 if x is None else len(x) for x in self.label_classes.values()]

        self.num_outputs = sum(self.label_counts)

        self.weighted_multilabel = False

        self.n_components = n_components

        self.resnet = ResNet(block=block, layers=layers, planes=planes, num_outputs=self.n_components,
                               include_features=include_features, features_only=features_only)

        self.mdn_head = GatedMultivariateMDNHead(self.num_outputs,
                                                 n_components=self.n_components,
                                                 init_mu=init_mu,
                                                 init_L=init_L)

        self.criterion = GatedMDNLoss(self.num_outputs)
        self.activations = None

        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.block = block
        self.planes = planes
        self.layers = layers
        self.features_only = features_only
        self.save_hyperparameters()

    def forward(self, x):
        features, outputs = self.resnet(x)
        return features, outputs

    def training_step(self, batch, batch_idx):
        return self._step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self._step('val', batch, on_step=False, on_epoch=True, sync_dist=True)

    def _step(self, phase, batch, **log_kwargs):
        images, labels = batch
        features, outputs = self.forward(images)
        pi = self.mdn_head(outputs)
        total_loss = self.criterion(pi, self.mdn_head.mu, self.mdn_head.sigma_L, labels)
        self.log(f'{phase}_loss', total_loss, **log_kwargs)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=self.gamma)
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch[0]
        features, outputs = self.forward(images)
        pi = self.mdn_head(outputs)
        return features, pi


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

label_choices = ['fcs', 'time', 'feed', 'starting_media', 'condition', 'sample'] + LMDataset.FC_COLS

def _add_training_args(parser):
    parser.add_argument("input", help="HDMF input file")
    parser.add_argument('label', type=str, nargs='+', choices=label_choices, help="the label to train with")

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

    if args.checkpoint is not None:
        model = LightningResNet.load_from_checkpoint(args.checkpoint)
    else:
        # This block expects 2D continuous outputs
        # This was implemented for
        model_kwargs = dict(lr=args.lr, step_size=args.step_size, gamma=args.gamma, block=args.block, planes=args.planes,
                            layers=args.layers)

        model_kwargs['n_components'] = 4

        dset = train_loader.dataset
        need_to_open = dset.table is None
        if need_to_open:
            dset.open()

        init_mu, init_L = init_mdn(np.asarray(dset.sample_labels).T)

        if need_to_open:
            dset.close()

        model_kwargs['init_mu'] = init_mu
        model_kwargs['init_L'] = init_L

        model = LightningResNet(train_loader.dataset.label_classes, **model_kwargs)

    return train_loader, val_loader, model


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", "--experiment", type=str, help="the experiment name")

    _add_training_args(parser)

    args = parser.parse_args(argv)

    logger = get_logger('debug')

    train_loader, val_loader, model = _get_loaders_and_model(args, logger=logger)

    timer = Timer()
    mod_sum = ModelSummary(max_depth=3)
    trainer = get_trainer(args, 'val_loss', extra_callbacks=[timer, mod_sum])
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
    parser.add_argument('label', type=str, nargs='+', choices=label_choices, help="the label to predict with")
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

    model = LightningResNet.load_from_checkpoint(args.checkpoint, label_classes=dataset.label_classes)

    targs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "limit_predict_batches": 100 if args.quick else None,
        "devices": 1
    }
    trainer = pl.Trainer(**targs)

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
    t.add_predicted_probability(predictions,
                                description="Component classification probabilities")

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
