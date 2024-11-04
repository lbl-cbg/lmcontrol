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

## add multilabels feature
## create a custom loss function usitng torch module
## create a weights dictionary and give 1000 for NMELoss

class LightningResNet(L.LightningModule):

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, label_classes, label_index_dict={'feed', 'time'}, label_counts={'time':1, 'feed':3}, num_outputs=1, lr=0.01, step_size=2, gamma=0.1, planes=[8, 16, 32, 64], layers=[1, 1, 1, 1], block=BasicBlock, return_embeddings=False, label_type='time'):
        super().__init__()

        weights = None 
        progress = True
        self.label_classes = label_classes
        self.label_index_dict = label_index_dict
        self.label_counts = label_counts
        self.loss_functions = {
            'time': nn.MSELoss(),
            'sample': nn.CrossEntropyLoss(),
            'condition': nn.CrossEntropyLoss(),
            'feed': nn.CrossEntropyLoss(),  
            'starting_media': nn.CrossEntropyLoss()  
        }
        self.loss_weights = {
            'time': 1e-4,
            'sample': 1,
            'condition': 1,
            'feed': 1,
            'starting_media': 1
        }
        
        if isinstance(label_type, str):
            label_type = [label_type]

        if label_type is not None:
            self.loss_functions = {key: self.loss_functions[key] for key in label_type if key in self.loss_functions}
            self.loss_weights = {key: self.loss_weights[key] for key in label_type if key in self.loss_weights}

        self.backbone = ResNet(label_type=label_type, block=block, layers=layers, planes=planes, num_outputs=num_outputs, return_embeddings=return_embeddings)
        if 'time' in label_type:
            self.backbone = nn.Sequential(self.backbone, nn.Softplus(), nn.Flatten(start_dim=1))

        self.save_hyperparameters()
        self.label_type = label_type
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.block = block
        self.planes = planes
        self.layers = layers
        
        if len(label_type) > 1:
            self.criteria = {key: self.loss_functions[key] for key in label_type} 
        else:
            self.criterion = self.loss_functions[label_type[0]]  

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        
        outputs_dict = {}
        start_idx = 0
        
        for key, count in self.label_counts.items():
            end_idx = start_idx + count
            outputs_dict[key] = outputs[:, start_idx:end_idx]
            start_idx = end_idx 

        total_loss = 0
        
        for key, output in outputs_dict.items():
            
            label_tensor = labels[self.label_index_dict[key]]
            if key == 'time':
                label_tensor = label_tensor.unsqueeze(1) 
            loss = self.loss_functions[key](output, label_tensor) * self.loss_weights[key]
            if key == 'time':
                label_tensor = label_tensor.squeeze(1)
            self.log(f'train_loss_{key}', loss, on_step=False, on_epoch=True)
            total_loss += loss
            
            if key != 'time':
                preds = torch.argmax(output, dim=1)
                acc = accuracy_score(label_tensor.cpu().numpy(), preds.cpu().numpy())
                self.log(f'train_accuracy_{key}', acc, on_step=False, on_epoch=True)

        self.log('total_train_loss', total_loss, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):  
        images, labels = batch
        outputs = self.forward(images)
        
        outputs_dict = {}
        start_idx = 0
        
        for key, count in self.label_counts.items():
            end_idx = start_idx + count
            outputs_dict[key] = outputs[:, start_idx:end_idx]
            start_idx = end_idx 

        total_loss = 0
        
        for key, output in outputs_dict.items():
            
            label_tensor = labels[self.label_index_dict[key]]
            
            if key == 'time':
                label_tensor = label_tensor.unsqueeze(1) 
            loss = self.loss_functions[key](output, label_tensor) * self.loss_weights[key]
            if key == 'time':
                label_tensor = label_tensor.squeeze(1)
            self.log(f'val_loss_{key}', loss, on_step=False, on_epoch=True)
            total_loss += loss
            
            if key != 'time':
                preds = torch.argmax(output, dim=1)
                acc = accuracy_score(label_tensor.cpu().numpy(), preds.cpu().numpy())
                self.log(f'val_accuracy_{key}', acc, on_step=False, on_epoch=True)

        self.log('total_val_loss', total_loss, on_step=False, on_epoch=True)
        return total_loss

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
    parser.add_argument('labels', type=str, nargs='+', choices=['time', 'feed', 'starting_media', 'condition', 'sample'], help="the label to train with")
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

    label_index_dict = {label: idx for idx, label in enumerate(train_dataset.label_type)}
    for label_type, index in label_index_dict.items():
        print(f"{label_type}: {index}")

    num_outputs = 0
    
    for label, idx in label_index_dict.items():
        if label == 'time':
            num_outputs += 1  
        else:
            current_labels = train_dataset.labels[idx]
            num_outputs += len(torch.unique(current_labels))

    label_counts = {}
    for label, idx in label_index_dict.items():
        if label == 'time':
            label_counts[label] = 1
        else:
            current_labels = train_dataset.labels[idx]
            label_counts[label] = len(torch.unique(current_labels))

    model = LightningResNet(label_classes=train_dataset.label_classes, label_index_dict=label_index_dict, label_counts=label_counts, num_outputs=num_outputs, lr=args.lr, step_size=args.step_size, gamma=args.gamma, block=args.block, planes=args.planes, layers=args.layers, return_embeddings=args.return_embeddings, label_type=args.labels)

    return train_loader, val_loader, model


def _get_trainer(args, trial=None):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    callbacks = []

    targs = dict(max_epochs=args.epochs, devices=1, accelerator=accelerator, check_val_every_n_epoch=4, callbacks=callbacks)
    
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
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_accuracy")) ## edit this for multilabels
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
    parser.add_argument('labels', type=str, nargs='+', choices=['time', 'feed', 'starting_media', 'condition', 'sample'], help="the label to predict with")  
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

    label_index_dict = {label: idx for idx, label in enumerate(predict_dataset.label_type)}
    for label_type, index in label_index_dict.items():
        print(f"{label_type}: {index}")
    
    num_outputs = 0
    
    for label, idx in label_index_dict.items():
        if label == 'time':
            num_outputs += 1  
        else:
            current_labels = predict_dataset.labels[idx]
            num_outputs += len(torch.unique(current_labels))

    label_counts = {}
    for label, idx in label_index_dict.items():
        if label == 'time':
            label_counts[label] = 1
        else:
            current_labels = predict_dataset.labels[idx]
            label_counts[label] = len(torch.unique(current_labels))

    true_labels = []
    for key in label_index_dict:
        index = label_index_dict[key] 
        true_labels.append(predict_dataset.labels[index])
  

    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=3)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)
    
    logger.info("Running predictions")
    predictions = trainer.predict(model, predict_loader)
    
    label_classes = model.label_classes
    predictions = torch.cat(predictions).numpy()
    
    out_data = dict(predictions=predictions, true_labels=true_labels, label_classes = label_classes)

    if args.return_embeddings:
        logger.info("Saving embeddings")
        out_data = dict(predictions=predictions)
        label_type = None
    else:
        logger.info("Classifier mode: Saving predictions using classification / regression")

        if true_labels is not None:
            out_data = {'predictions': predictions, 'true_labels': true_labels, 'label_classes': label_classes}   
            for key in label_index_dict:
                index = label_index_dict[key]
                
                if key == 'time':
                    time_predictions = predictions[:, index]  
                    true_time_labels = true_labels[index]     

                    mse = mean_squared_error(true_time_labels, time_predictions)
                    mae = mean_absolute_error(true_time_labels, time_predictions)
                    r2 = r2_score(true_time_labels, time_predictions)

                    logger.info(f"Mode: Regression for {key}")
                    logger.info(f"Mean Squared Error: {mse:.4f}")
                    logger.info(f"Mean Absolute Error: {mae:.4f}")
                    logger.info(f"R² Score: {r2:.4f}")

                    if args.save_residuals:
                        logger.info("Calculating and saving residuals")
                        residuals = true_time_labels - time_predictions
                        out_data['residuals'] = residuals

                    out_data['true_labels_time'] = true_time_labels
                    out_data['pred_labels_time'] = time_predictions

                else:
                    logger.info(f"Mode: Classification for {key}")

                    num_classes = label_counts[key]

                    label_predictions = predictions[:, index:index + num_classes]  
                    pred_labels = np.argmax(label_predictions, axis=1)

                    accuracy = accuracy_score(true_labels[index], pred_labels)  
                    precision = precision_score(true_labels[index], pred_labels, average='weighted')
                    recall = recall_score(true_labels[index], pred_labels, average='weighted')
                    conf_matrix = confusion_matrix(true_labels[index], pred_labels)

                    logger.info(f"Accuracy: {accuracy:.4f}")
                    logger.info(f"Precision: {precision:.4f}")
                    logger.info(f"Recall: {recall:.4f}")
                    logger.info(f"Confusion Matrix:\n{conf_matrix}")

                    correct_incorrect = np.where(pred_labels == true_labels[index], 1, 0)
                    out_data['correct_incorrect'] = correct_incorrect

                    out_data[f'true_labels_{key}'] = true_labels[index]
                    out_data[f'pred_labels_{key}'] = pred_labels
                    out_data[f'label_predictions_{key}'] = label_predictions  

        else:
            out_data = dict(predictions=predictions)

    if not args.pred_only:
        dset = predict_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        for key in dset.metadata:
            out_data[key] = np.asarray(dset.metadata[key])
        # for i, k in enumerate(dset.label_type):                   # no need for this as we are anyways obtaining the key wise labels 
        #     out_data[k + "_labels"] = np.asarray(dset.labels[i])

    np.savez(args.output_npz, **out_data)

    logger.info(f"Results saved to {args.output_npz}")


if __name__ == '__main__':
    predict()
