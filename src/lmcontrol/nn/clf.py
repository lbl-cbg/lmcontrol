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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Type, Union, List, Optional, Callable, Any
from optuna.integration import PyTorchLightningPruningCallback

from ..utils import get_logger
from ..data_utils import load_npzs, encode_labels
from .dataset import LMDataset, get_transforms as _get_transforms
from .resnet import ResNet, add_args as add_resnet_args
from .utils import get_loaders


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


class LightningResNet(L.LightningModule):

    val_metric = "total_val_loss"
    train_metric = "total_train_loss"

    loss_functions = {
        'time': nn.MSELoss(),
        'sample': CrossEntropy(),
        'condition': CrossEntropy(),
        'feed': CrossEntropy(),
        'starting_media': CrossEntropy()
    }


    def __init__(self, label_classes, lr=0.01, step_size=2, gamma=0.1, planes=[8, 16, 32, 64],
                 layers=[1, 1, 1, 1], block=BasicBlock, return_embeddings=False, time_weight=1e-3):
        super().__init__()

        self.loss_weights = {
            'time': time_weight,
            'sample': 0,
            'condition': 0,
            'feed': 1-time_weight,
            'starting_media': 0
        }


        self.label_classes = label_classes

        self.activations = [nn.Sequential(nn.Softplus(), nn.Flatten(start_dim=0)) if LMDataset.is_regression(l) else nn.Softmax(dim=1)
                            for l in self.label_classes]

        self.label_counts = [1 if x is None else len(x) for x in self.label_classes.values()]

        self.num_outputs = sum(self.label_counts)

        self.criterion = MultiLabelLoss([self.loss_functions[l] for l in self.label_classes],
                                        weights=[self.loss_weights[l] for l in self.label_classes])


        self.backbone = ResNet(block=block, layers=layers, planes=planes, num_outputs=self.num_outputs,
                               return_embeddings=return_embeddings)

        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.block = block
        self.planes = planes
        self.layers = layers
        self.return_embeddings = return_embeddings
        self.save_hyperparameters()

    def forward(self, x):
        outputs = self.backbone(x)
        if not self.return_embeddings:
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
        loss_components = self.criterion(outputs, labels)
        self._score("train", outputs, loss_components, labels)
        total_loss = sum(loss_components)
        self.log(self.train_metric, total_loss, on_step=False, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss_components = self.criterion(outputs, labels)
        self._score("val", outputs, loss_components, labels)
        total_loss = sum(loss_components)
        self.log(self.val_metric, total_loss, on_step=False, on_epoch=True)
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
    parser.add_argument("input", type=str, help="Training image data")
    parser.add_argument('label', type=str, nargs='+', choices=['time', 'feed', 'starting_media', 'condition', 'sample'], help="the label to train with")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--validation", type=str, nargs='+', help="directories containing validation data")
    grp.add_argument("--val_frac", type=float, default=None, help="Part of data to use for training (between 0 and 1)")
    parser.add_argument("--seed", type=int, default=None, help="seed for training-validation-test split")

    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-n", "--n_samples", type=int, help="number of samples to use from each NPZ", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)
    parser.add_argument("--wandb", action='store_false', default=True, help="provide this flag to stop wandb")
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--step_size", type=int, help="step size for learning rate scheduler", default=10)
    parser.add_argument("--gamma", type=float, help="gamma for learning rate scheduler", default=0.1)
    parser.add_argument("--batch_size", type=int, help="batch size for training and validation", default=32)
    parser.add_argument("--time_weight", type=float, help="loss function weight for time", default=0.001)

    add_resnet_args(parser)

    parser.add_argument("--return_embeddings", action='store_true', default=False, help="return ResNet features, rather than final output")


def _get_loaders_and_model(args,  inference=False, logger=None):
    train_transform = _get_transforms('float', 'norm', 'blur','rotate', 'crop', 'hflip', 'vflip', 'noise', 'rgb')
    val_transform = _get_transforms('float', 'norm', 'blur','rotate', 'crop', 'hflip', 'vflip', 'noise', 'rgb')

    train_loader, val_loader = get_loaders(args,
                                           inference=inference,
                                           train_tfm=train_transform,
                                           val_tfm=val_transform,
                                           exp_split=True,
                                           return_labels=True)


    model = LightningResNet(train_dataset.label_classes, lr=args.lr, step_size=args.step_size, gamma=args.gamma,
                            block=args.block, planes=args.planes, layers=args.layers, return_embeddings=args.return_embeddings, time_weight=args.time_weight)

    return train_loader, val_loader, model


def _get_trainer(args, trial=None):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks = []

    targs = dict(num_nodes=args.num_nodes, max_epochs=args.epochs, devices=args.devices,
                 accelerator="gpu" if args.devices > 0 else "cpu", check_val_every_n_epoch=1, callbacks=callbacks)

    if args.devices > 0:
        torch.set_float32_matmul_precision('medium')

    if args.devices > 1:  # If using multiple GPUs, use Distributed Data Parallel (DDP)
        targs['strategy'] = DDPStrategy(find_unused_parameters=True)

    # should we use r2 score and val_accuracy for measurement
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.outdir,
        filename="checkpoint-{epoch:02d}-{validation_ncs:.4f}",
        save_top_k=3,
        monitor="validation_ncs",
        mode="min",
        save_last=True,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint_callback)

    early_stopping_callback = EarlyStopping(
        monitor="validation_ncs",
        patience=5,
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

    if args.debug:
        targs['limit_train_batches'] = 100
        targs['limit_val_batches'] = 10

    return L.Trainer(**targs)


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", "--experiment", type=str, help="the experiment name")

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
    trainer = _get_trainer(args, trial=trial)
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
    parser.add_argument('label', type=str, nargs='+', choices=['time', 'feed', 'starting_media', 'condition', 'sample'], help="the label to predict with")
    parser.add_argument("--prediction", type=str, nargs='+', required=True, help="directories containing prediction data")
    parser.add_argument("-c", "--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-o", "--output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")
    parser.add_argument("-n", "--n_samples", type=int, help="number of samples to use from each class", default=None)
    parser.add_argument("--return_embeddings", action='store_true', default=False, help="provide this if you don't want classifier, helpful in embeddings stuff")
    parser.add_argument("--save_residuals", action='store_true', default=False, help="provide this if you want to store the residual values")

    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = _get_transforms('float', 'norm', 'crop', 'rgb')

    predict_files = args.prediction
    n = args.n_samples


    logger.info(f"Loading prediction data: {len(predict_files)} files")
    predict_dataset = LMDataset(predict_files, transform=transform, logger=logger, return_labels=True, label=args.label, n_samples=n)

    model = LightningResNet.load_from_checkpoint(args.checkpoint, label_classes=predict_dataset.label_classes, return_embeddings=args.return_embeddings)

    for i in range(len(predict_dataset.sample_labels)):
        current_labels = predict_dataset.sample_labels[i]
        logger.info(predict_dataset.label[i] + " - " + str(torch.unique(current_labels)))


    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=3)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)

    logger.info("Running predictions")


    if not args.return_embeddings:
        predictions = [torch.cat(l).numpy() for l in zip(*trainer.predict(model, predict_loader))]
        true_labels = predict_dataset.sample_labels
        label_classes = model.label_classes
        out_data = dict()

        for pred, true, key in zip(predictions, true_labels, label_classes):
            true = true.numpy()

            out_data[key] = {
                'output': pred,
                'labels': true,
            }

            if LMDataset.is_regression(key):

                mse = mean_squared_error(true, pred)
                mae = mean_absolute_error(true, pred)
                r2 = r2_score(true, pred)

                logger.info(f"Mode: Regression for {key}")
                logger.info(f"Mean Squared Error: {mse:.4f}")
                logger.info(f"Mean Absolute Error: {mae:.4f}")
                logger.info(f"R² Score: {r2:.4f}")

                if args.save_residuals:
                    logger.info("Calculating and saving residuals")
                    residuals = true - pred
                    out_data[key]['residuals'] = residuals

            else:
                logger.info(f"Mode: Classification for {key}")

                pred_labels = np.argmax(pred, axis=1)

                accuracy = accuracy_score(true, pred_labels)
                precision = precision_score(true, pred_labels, average='weighted')
                recall = recall_score(true, pred_labels, average='weighted')
                conf_matrix = confusion_matrix(true, pred_labels)

                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"Confusion Matrix:\n{conf_matrix}")
                out_data[key]['pred'] = pred_labels
                out_data[key]['classes'] = label_classes[key]

    else:
        predictions = trainer.predict(model, predict_loader)
        label_classes = model.label_classes
        out_data = dict()
        true_labels = predict_dataset.sample_labels

        for key in label_classes:

            predictions = np.concatenate(predictions, axis=0)
            out_data[key] = {
                'output': predictions,
                'labels': true_labels
            }


    if not args.pred_only:
        dset = predict_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        out_data['metadata'] = {key: np.asarray(dset.metadata[key]) for key in dset.metadata}

    np.savez(args.output_npz, **out_data)

    logger.info(f"Results saved to {args.output_npz}")


if __name__ == '__main__':
    predict()
