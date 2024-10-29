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
from lmcontrol.nn.resnet import ResNet


class LightningResNet(L.LightningModule):

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, num_outputs=1, lr=0.01, step_size=2, gamma=0.1, planes=[8, 16, 32, 64], layers=[1, 1, 1, 1], block=BasicBlock, return_embeddings=False, label_type='time'):
        super().__init__()

        weights = None 
        progress = True

        self.loss_functions = {
            'time': nn.MSELoss(),
            'sample': nn.CrossEntropyLoss(),
            'condition': nn.CrossEntropyLoss(),
            'feed': nn.CrossEntropyLoss(),  
            'starting media': nn.CrossEntropyLoss()  
        }
        if label_type not in self.loss_functions:
            raise ValueError(f"Unknown label type: {label_type}. Expected one of {list(self.loss_functions.keys())}")
        
        self.backbone = ResNet(label_type=label_type, block=block, layers=layers, planes=planes, num_outputs=num_outputs, return_embeddings=return_embeddings)
        if label_type == 'time':
            self.backbone = nn.Sequential(self.backbone, nn.Softplus(), nn.Flatten(start_dim=1))
        self.criterion = self.loss_functions[label_type]
        self.save_hyperparameters()
        self.label_type = label_type
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

        if self.label_type != 'time':
            preds = torch.argmax(outputs, dim=1)
            acc = accuracy_score(labels[:, 0].cpu().numpy(), preds.cpu().numpy())
            self.log('train_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_loss', loss, on_epoch=True)
        elif self.label_type == 'time':
            self.log('train_loss', loss, on_epoch=True)
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels[:, 0]) 

        if self.label_type != 'time':
            preds = torch.argmax(outputs, dim=1)
            acc = accuracy_score(labels[:, 0].cpu().numpy(), preds.cpu().numpy())
            self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_loss', loss, on_epoch=True)
        elif self.label_type == 'time':
            self.log('val_loss', loss, on_epoch=True)
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=self.gamma)
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  
        x = batch[0]
        return self.forward(x)


# def get_transform():
#     transform = _get_transforms('float', 'norm','blur','rotate', 'crop','hflip', 'vflip', 'noise', 'rgb')
#     return transform

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
    parser.add_argument('labels', type=str, choices=['time', 'feed', 'starting_media', 'condition', 'sample'], help="the label to train with")
    parser.add_argument("--training", type=str, nargs='+', required=True, help="directories containing training data")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--validation", type=str, nargs='+', help="directories containing validation data")
    grp.add_argument("--val_frac", type=float, default=None, help="Part of data to use for training (between 0 and 1)")

    parser.add_argument("--seed", type=int)
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-n", "--n_samples", type=int, help="number of samples to use from each NPZ", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)
    parser.add_argument("-stop_wandb", "--stop_wandb", action='store_false', default=True, help="provide this flag to stop wandb")
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)                               
    parser.add_argument("--step_size", type=int, help="step size for learning rate scheduler", default=10)
    parser.add_argument("--gamma", type=float, help="gamma for learning rate scheduler", default=0.1)
    parser.add_argument("--batch_size", type=int, help="batch size for training and validation", default=32)
    parser.add_argument("--block", type=get_block, choices=['BasicBlock', 'Bottleneck'], help="type of block to use in the model", default='Bottleneck')
    parser.add_argument("--planes", type=get_planes, choices=['3', '4'], help="list of number of planes for each layer", default='4')
    parser.add_argument("--layers", type=get_layers, choices=['1', '2', '3', '4'], help="list of number of layers in each stage", default='4')
    parser.add_argument("-save_emb", "--return_embeddings", action='store_true', default=False, help="saves embeddings, used for plotly/dash")

def _get_loaders_and_model(args,  logger=None):
    transform_train = _get_transforms('float', 'norm','blur','rotate', 'crop','hflip', 'vflip', 'noise', 'rgb') 
    transform_val = _get_transforms('float', 'norm','blur','rotate', 'crop','hflip', 'vflip', 'noise', 'rgb')                                                        

    if args.val_frac:
        split_files = args.training
        
        n = args.n_samples

        if logger is None:
            logger = get_logger("critical")

        logger.info(f"Loading training data from: {len(split_files)} files")
        train_dataset = LMDataset(split_files, transform=transform_train, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings, split='train', val_size=args.val_frac, seed=args.seed)

        for i in range(train_dataset.labels.shape[1]):
                logger.info(train_dataset.label_type[i] + " - " + str(torch.unique(train_dataset.labels[:, i])) + str(train_dataset.label_classes))


        logger.info(f"Loading validation data from: {len(split_files)} files")
        val_dataset = LMDataset(split_files, transform=transform_val, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings, split='validate', val_size=args.val_frac, seed=args.seed)
        for i in range(val_dataset.labels.shape[1]):
            logger.info(val_dataset.label_type[i] + " - " + str(torch.unique(val_dataset.labels[:, i])) + str(val_dataset.label_classes))

    elif args.validation:
        train_files = args.training
        val_files = args.validation

        n = args.n_samples

        if logger is None:
            logger = get_logger("critical")

        logger.info(f"Loading training data: {len(train_files)} files")
        train_dataset = LMDataset(train_files, transform=transform_train, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings)

        for i in range(train_dataset.labels.shape[1]):
                logger.info(train_dataset.label_type[i] + " - " + str(torch.unique(train_dataset.labels[:, i])) + str(train_dataset.label_classes))


        logger.info(f"Loading validation data: {len(val_files)} files")
        val_dataset = LMDataset(val_files, transform=transform_val, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings)
        for i in range(val_dataset.labels.shape[1]):
            logger.info(val_dataset.label_type[i] + " - " + str(torch.unique(val_dataset.labels[:, i])) + str(val_dataset.label_classes))

    else:
        print("You must specify --validation or --val_frac", file=sys.stderr)
        exit(1)

    num_workers = 0 if args.debug else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)

    if args.labels != 'time':
        num_outputs = len(train_dataset.label_classes[0])
    elif args.labels == 'time':
        num_outputs = 1
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    model = LightningResNet(num_outputs=num_outputs,lr=args.lr, step_size=args.step_size, gamma=args.gamma, block=args.block, planes=args.planes, layers=args.layers, return_embeddings=args.return_embeddings, label_type=args.labels)

    return train_loader, val_loader, model


def _get_trainer(args, trial=None): 
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks = []

    targs = dict(max_epochs=args.epochs, devices=1, accelerator=accelerator, check_val_every_n_epoch=4, callbacks=callbacks)
    
    if args.labels != 'time':
        if args.early_stopping:
            early_stopping = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.0001,
            patience=10,
            verbose=False,
            mode="max"
            )
            callbacks.append(early_stopping)

        if args.checkpoint:
            checkpoint_callback = ModelCheckpoint(
                dirpath=args.checkpoint,  
                filename="checkpoint-{epoch:02d}-{val_accuracy:.4f}",  
                save_top_k=3, 
                monitor="val_accuracy", 
                mode="max"  
            )
            callbacks.append(checkpoint_callback)
    elif args.labels == 'time':
        if args.early_stopping:
            early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
            verbose=False,
            mode="min"
            )
            callbacks.append(early_stopping)

        if args.checkpoint:
            checkpoint_callback = ModelCheckpoint(
                dirpath=args.checkpoint,  
                filename="checkpoint-{epoch:02d}-{val_loss:.4f}",  
                save_top_k=3, 
                monitor="val_loss", 
                mode="min"  
            )
            callbacks.append(checkpoint_callback)
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")
        
    if trial is not None :   # if 'trial' is passed in, assume we are using Optuna to do HPO        
        targs['logger'] = CSVLogger(args.outdir, name=args.experiment)
        if args.pruning:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_accuracy"))
    else:
        if args.stop_wandb:
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

    args.outdir = os.path.join(args.working_dir, "logs")   
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
        os.makedirs(args.working_dir, exist_ok=True) 
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
    parser.add_argument('labels', type=str, help="the label to predict with")  
    parser.add_argument("--prediction", type=str, nargs='+', required=True, help="directories containing prediction data")
    parser.add_argument("-c", "--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-o", "--output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")
    parser.add_argument("-n", "--n_samples", type=int, help="number of samples to use from each class", default=None)
    parser.add_argument("-save_emb", "--return_embeddings", action='store_true', default=False, help="provide this if you don't want classifier, helpful in embeddings stuff")
    parser.add_argument("-save_residuals", "--save_residuals", action='store_true', default=False, help="provide this if you want to store the residual values")

    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = _get_transforms('float', 'norm', 'crop', 'rgb')

    predict_files = args.prediction
    n = args.n_samples

    logger.info(f"Loading prediction data: {len(predict_files)} files")
    predict_dataset = LMDataset(predict_files, transform=transform, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings)
    for i in range(predict_dataset.labels.shape[1]):
        logger.info(predict_dataset.label_type[i] + " - " + str(torch.unique(predict_dataset.labels[:, i])) + str(predict_dataset.label_classes))

    true_labels = predict_dataset.labels[:, 0]

    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=3)

    model = LightningResNet.load_from_checkpoint(args.checkpoint, return_embeddings=args.return_embeddings)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)
    
    logger.info("Running predictions")
    predictions = trainer.predict(model, predict_loader)
    predictions = torch.cat(predictions).numpy()

    if args.return_embeddings:
        logger.info("Saving embeddings")
        out_data = dict(predictions=predictions)
        label_type = None
    else:
        logger.info("Classifier mode: Saving predictions using classification / regression")

        if true_labels is not None:
            if args.labels != 'time':
                    logger.info("mode:classification")
                    pred_labels = np.argmax(predictions, axis=1)

                    accuracy = accuracy_score(true_labels, pred_labels)
                    precision = precision_score(true_labels, pred_labels, average='weighted')
                    recall = recall_score(true_labels, pred_labels, average='weighted')
                    conf_matrix = confusion_matrix(true_labels, pred_labels)

                    logger.info(f"Accuracy: {accuracy:.4f}")
                    logger.info(f"Precision: {precision:.4f}")
                    logger.info(f"Recall: {recall:.4f}")
                    logger.info(f"Confusion Matrix:\n{conf_matrix}")

                    out_data = dict(predictions=predictions, true_labels=true_labels, pred_labels=pred_labels)
                    correct_incorrect = np.where(pred_labels == true_labels, 1, 0)
                    out_data['correct_incorrect'] = correct_incorrect

            elif args.labels == 'time': 
                    logger.info("mode:regression")
                    pred_labels = np.argmax(predictions, axis=0)
                    pred_values = predictions.flatten()

                    mse = mean_squared_error(true_labels, pred_values)
                    mae = mean_absolute_error(true_labels, pred_values)
                    r2 = r2_score(true_labels, pred_values)

                    logger.info(f"Mean Squared Error: {mse:.4f}")
                    logger.info(f"Mean Absolute Error: {mae:.4f}")
                    logger.info(f"R² Score: {r2:.4f}")

                    if args.save_residuals:
                        logger.info("Calculating and saving residuals")
                        residuals = true_labels - pred_values
                        out_data = dict(predictions=predictions, true_values=true_labels, pred_values=pred_values, residuals=residuals)
                    else:
                        out_data = dict(predictions=predictions, true_labels=true_labels, pred_labels=pred_values)
            else:
                raise ValueError("task_type must be either 'classification' or 'regression'")

        else:
            out_data = dict(predictions=predictions, pred_labels=pred_values)

    if not args.pred_only:
        dset = predict_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        for i, k in enumerate(dset.label_type):
            out_data[k + "_labels"] = np.asarray(dset.labels[:, i])

    np.savez(args.output_npz, **out_data)

    logger.info(f"Results saved to {args.output_npz}")


if __name__ == '__main__':
    predict()
