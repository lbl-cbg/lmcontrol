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

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    loss_functions = {
        'time': nn.MSELoss(),
        'sample': CrossEntropy(),
        'conditions': CrossEntropy(),
        'feed': CrossEntropy(),
        'starting_media': CrossEntropy()
    }


    def __init__(self, label_classes, lr=0.01, step_size=2, gamma=0.1, planes=[8, 16, 32, 64],
                 layers=[1, 1, 1, 1], block=BasicBlock, return_embeddings=False, time_weight=1e-3): # weight2=1, weight3=1, weight4=1, weight5=1
        super().__init__()

        self.loss_weights = {
            'time': time_weight,  
            'sample': 0,
            'conditions': 0,
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
        self.save_hyperparameters()

    def forward(self, x):
        outputs = self.backbone(x)
        ret = list()
        start_idx = 0
        for act, count in zip(self.activations, self.label_counts):
            end_idx = start_idx + count
            ret.append(act(outputs[:, start_idx:end_idx]))
            start_idx = end_idx
        return tuple(ret)

    
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
        self.log('total_train_loss', total_loss, on_step=False, on_epoch=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss_components = self.criterion(outputs, labels)
        
        self._score("val", outputs, loss_components, labels)
        
        total_loss = sum(loss_components)
        self.log('total_val_loss', total_loss, on_step=False, on_epoch=True)
    
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
    parser.add_argument('labels', type=str, nargs='+', choices=['time', 'feed', 'starting_media', 'conditions', 'sample'], help="the label to train with")
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
    parser.add_argument("--time_weight", type=float, help="loss function weight for time", default=0.001)

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

        for i in range(len(train_dataset.labels)):
                current_labels = train_dataset.labels[i]
                logger.info(train_dataset.label_type[i] + " - " + str(torch.unique(current_labels)))


        logger.info(f"Loading validation data from: {len(split_files)} files")
        val_dataset = LMDataset(split_files, label_classes=train_dataset.label_classes, transform=transform_val, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings, split='validate', val_size=args.val_frac, seed=args.seed)
        for i in range(len(val_dataset.labels)):
            current_labels = train_dataset.labels[i]
            logger.info(val_dataset.label_type[i] + " - " + str(torch.unique(current_labels)))

    elif args.validation:
        train_files = args.training
        val_files = args.validation

        n = args.n_samples

        if logger is None:
            logger = get_logger("critical")

        logger.info(f"Loading training data: {len(train_files)} files")
        train_dataset = LMDataset(train_files, transform=transform_train, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings)

        for i in range(len(train_dataset.labels)):
                current_labels = train_dataset.labels[i]
                logger.info(train_dataset.label_type[i] + " - " + str(torch.unique(current_labels)))

        logger.info(f"Loading validation data: {len(val_files)} files")
        val_dataset = LMDataset(val_files, label_classes=train_dataset.label_classes, transform=transform_val, logger=logger, return_labels=True, label_type=args.labels, n_samples=n, return_embeddings=args.return_embeddings)
        for i in range(len(val_dataset.labels)):
            current_labels = train_dataset.labels[i]
            logger.info(val_dataset.label_type[i] + " - " + str(torch.unique(current_labels)))

    else:
        print("You must specify --validation or --val_frac", file=sys.stderr)
        exit(1)

    num_workers = 0 if args.debug else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)


    model = LightningResNet(train_dataset.label_classes, lr=args.lr, step_size=args.step_size, gamma=args.gamma,
                            block=args.block, planes=args.planes, layers=args.layers, return_embeddings=args.return_embeddings, time_weight=args.time_weight)

    return train_loader, val_loader, model


def _get_trainer(args, trial=None):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks = []

    targs = dict(max_epochs=args.epochs, devices=1, accelerator=accelerator, check_val_every_n_epoch=4, callbacks=callbacks)

    # should we use r2 score and val_accuracy for measurement 
    if args.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint,
            filename="checkpoint-{epoch:02d}-{total_val_loss:.4f}",
            save_top_k=3,
            monitor="total_val_loss",
            mode="min"
        )
        callbacks.append(checkpoint_callback)


    if trial is not None :   # if 'trial' is passed in, assume we are using Optuna to do HPO
        targs['logger'] = CSVLogger(args.outdir, name=args.experiment)
        if args.pruning:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="combined_metric"))

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
    # args.batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    # args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # args.step_size = trial.suggest_int('step_size', 5, 15, step=5)
    # args.gamma = trial.suggest_float('gamma', 0.1, 0.5)
    args.time_weight = trial.suggest_float('time_weight', 1e-5, 1.0, log=True)
    # args.weight2 = trial.suggest_float('weight2', 1e-5, 1.0, log=True)
    # args.weight3 = trial.suggest_float('weight3', 1e-5, 1.0, log=True)
    # args.weight4 = trial.suggest_float('weight4', 1e-5, 1.0, log=True)
    # args.weight5 = trial.suggest_float('weight5', 1e-5, 1.0, log=True)
    
    # block_type = trial.suggest_categorical('block_type', ['BasicBlock', 'Bottleneck'])
    # args.block = get_block(block_type)

    # p = trial.suggest_categorical('planes', ['3', '4'])
    # args.planes = get_planes(p)

    # l = trial.suggest_categorical('layers', ['1', '2', '3', '4'])
    # args.layers = get_layers(l)

    args.outdir = os.path.join(args.working_dir, "logs")
    args.experiment = f"trial_{trial.number:04d}"
    
    train_loader, val_loader, model = _get_loaders_and_model(args)
    trainer = _get_trainer(args, trial=trial)
    trainer.fit(model, train_loader, val_loader)

    val_accuracy = trainer.callback_metrics.get("val_mean_accuracy", None)
    val_r2 = trainer.callback_metrics.get("val_mean_r2", None)
    
    # combined_metric = 0.5 * (val_accuracy if val_accuracy is not None else 0) + \
    #                   0.5 * (val_r2 if val_r2 is not None else 0)
                      
    # return combined_metric
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
    parser.add_argument('labels', type=str, nargs='+', choices=['time', 'feed', 'starting_media', 'conditions', 'sample'], help="the label to predict with")
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
    
    model = LightningResNet.load_from_checkpoint(args.checkpoint, label_classes=predict_dataset.label_classes, return_embeddings=args.return_embeddings)

    for i in range(len(predict_dataset.labels)):
        current_labels = predict_dataset.labels[i]
        logger.info(predict_dataset.label_type[i] + " - " + str(torch.unique(current_labels)))
        

    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=3)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)

    logger.info("Running predictions")


    if not args.return_embeddings:
        predictions = [torch.cat(l).numpy() for l in zip(*trainer.predict(model, predict_loader))]
        true_labels = predict_dataset.labels
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
        out_data = dict(embeddings=predictions)
    
    
    if not args.pred_only:
        dset = predict_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        out_data['metadata'] = {key: np.asarray(dset.metadata[key]) for key in dset.metadata}

    np.savez(args.output_npz, **out_data)

    logger.info(f"Results saved to {args.output_npz}")


if __name__ == '__main__':
    predict()
