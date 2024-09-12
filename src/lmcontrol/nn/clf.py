import argparse
import glob
import lightning as L
import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn
import torch
import glob
import numpy as np
import argparse
import lightning as L
from ..utils import get_logger
from .dataset import get_transforms as _get_transforms
import wandb
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import argparse
import os
import matplotlib.pyplot as plt

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

from torchvision.models.resnet import BasicBlock, _resnet  ####

from typing import Type, Union, List, Optional, Callable, Any

import tempfile
import torch.nn.functional as F
from filelock import FileLock
from torch.utils.data import DataLoader, random_split

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ray.train import RunConfig, ScalingConfig, CheckpointConfig

import pytorch_lightning as pl  ## REMOVE 

def resnet10(*, weights = None, progress = True, **kwargs) :
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

    return _resnet(BasicBlock, [1, 1, 1, 1], weights, progress, **kwargs)

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


class ResNet(L.LightningModule):

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, num_classes, lr=0.01, step_size=2, gamma=0.1 ): ###CHANGED 
        super().__init__()
        #self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone = resnet10()
        n_features = 512 
        self.backbone.fc = nn.Linear(n_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters() #this is an important part, we didnt add it before. STUDY WHY THIS MATTERED 

        #self.foo = 'bar'  # this creates an attribute called "foo" on the instance. the attribute's value is "bar"
        self.lr = lr # even if we mention lr here, we obtain the passed value and not the default value mentioned above 
        self.step_size = step_size
        self.gamma = gamma


    """
    resnet = ResNet(4, l_r=0.0001, step_size=7,gamma=1) #Temporary for learning purposes 

    ...../

    ...

    ..

    scheduler, optimizer = resnet.configure_optimizers()
    """

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):  #Added this part, read the pytorch documentation to understand how its used 
        x = batch[0]
        return self.forward(x).flatten(start_dim=1)    


def get_transform():
    transform = _get_transforms('float', 'norm','blur','rotate', 'crop','hflip', 'vflip', 'noise', 'rgb')
    return transform


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument('labels', type=str, help="the label to train with")
    parser.add_argument("--training", type=str, nargs='+', required=True, help="directories containing training data")
    parser.add_argument("--validation", type=str, nargs='+', required=True, help="directories containing validation data")
    parser.add_argument("-c", "--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-n", "--data_size", type=int, help="number of samples to use from each class", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--step_size", type=int, help="step size for learning rate scheduler", default=10)
    parser.add_argument("--gamma", type=float, help="gamma for learning rate scheduler", default=0.1)
    parser.add_argument("--batch_size", type=int, help="batch size for training and validation", default=32)
    parser.add_argument("--tune", action='store_true', help="perform hyperparameter tuning", default=False)
    parser.add_argument("--num_samples", type=int, help="number of samples for tuning", default=5)
    parser.add_argument("--gpus_per_trial", type=int, help="number of GPUs per trial for tuning", default=1)
    parser.add_argument("--max_num_epochs", type=int, help="maximum number of epochs for tuning", default=10)

    args = parser.parse_args(argv)
    logger = get_logger('info')

    transform = get_transform()

    train_files = args.training
    val_files = args.validation
    n = args.data_size

    logger.info(f"Loading training data: {len(train_files)} files")
    train_dataset = LMDataset(train_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels, n=n)
    for i in range(train_dataset.labels.shape[1]):
        logger.info(train_dataset.label_types[i] + " - " + str(torch.unique(train_dataset.labels[:, i])) + str(train_dataset.label_classes))
    
    logger.info(f"Loading validation data: {len(val_files)} files")
    val_dataset = LMDataset(val_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels, n=n)
    for i in range(val_dataset.labels.shape[1]):
        logger.info(val_dataset.label_types[i] + " - " + str(torch.unique(val_dataset.labels[:, i])) + str(val_dataset.label_classes))

    num_workers = 0 if args.debug else 4

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "step_size": tune.choice([5, 10, 15]),
        "gamma": tune.loguniform(0.1, 0.9),
        "batch_size": tune.choice([32, 64, 128, 256])   
    }

    def train_model(config, hpo=False):
        model = ResNet(num_classes=len(train_dataset.label_classes[0]), lr=config["lr"], step_size=config["step_size"], gamma=config["gamma"])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=num_workers)

        callbacks = []
        trainer_args = dict(callbacks=callbacks,
                            max_epochs=config.get('epochs', 1),
                            check_val_every_n_epoch=config.get('check_val_every_n_epochs', 4),
                            devices = config.get("devices", 1),
                            accelerator=config.get('accelerator', "gpu" if torch.cuda.is_available() else "cpu")
                        )

        if hpo:
            trainer_args['strategy'] = RayDDPStrategy()   # keep in case we want to do parallel training
            callbacks.append(RayTrainReportCallback())
            trainer_args['plugins'] = [RayLightningEnvironment()]
            trainer_args['enable_progress_bar'] = False
        else:
            trainer_args['logger'] = WandbLogger(project='SX_HTY_Run1', log_model=True)
            wandb.init(project="SX_HTY_Run1")
            callbacks.append(ModelCheckpoint(
                monitor='val_accuracy',
                dirpath=config.get('outdir', '.'),
                save_top_k=1,
                mode='max'
            ))
            if config.get('early_stopping', False):
                early_stopping = EarlyStopping(
                    monitor="val_accuracy",
                    min_delta=0.000001,
                    patience=10,
                    verbose=False,
                    mode="max"
                )
                callbacks.append(early_stopping)
        
        trainer = L.Trainer(**trainer_args)
        
        if hpo:
            trainer = prepare_trainer(trainer)
        
        trainer.fit(model, train_loader, val_loader)

    if args.tune:

        print("Starting tuning...")

        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )

        search_space = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "step_size": tune.choice([5, 10, 15]),
            "gamma": tune.loguniform(0.1, 0.9),
            "batch_size": tune.choice([32, 64, 128, 256])
        }

        num_epochs = 20
        num_samples = 50

        scaling_config = ScalingConfig(
            num_workers=4, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
        )

        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val_accuracy",
                checkpoint_score_order="max",
            ),
        )

        from ray.train.torch import TorchTrainer
        from functools import partial

        train_func = partial(train_model, hpo=True)

        # Define a TorchTrainer without hyper-parameters for Tuner
        ray_trainer = TorchTrainer(
            train_func,
            scaling_config=scaling_config,
            #train_loop_config=config,  #from https://discuss.ray.io/t/raytrainreportcallback-error-using-in-pytorch-lightning/12243/4
            run_config=run_config,
        )

        def tune_mnist_asha(num_samples=10):
            scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

            tuner = tune.Tuner(
                ray_trainer,
                param_space={"train_loop_config": search_space},
                tune_config=tune.TuneConfig(
                    metric="val_accuracy",
                    mode="max",
                    num_samples=num_samples,
                    scheduler=scheduler,
                ),
            )
            return tuner.fit()

        results = tune_mnist_asha(num_samples=num_samples)


    else:
        train_model(vars(args), hpo=False)



def predict(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, help="the label to predict with") 
    parser.add_argument("--prediction", type=str, nargs='+', required=True, help="directories containing prediction data")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")      
    parser.add_argument("-o","--output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")
    parser.add_argument("-n", "--data_size", type=int, help="number of samples to use from each class's HT_X", default=None)

    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = get_transform()  

    predict_files = args.prediction
    n = args.data_size

    logger.info(f"Loading predicting data: {len(predict_files)} files")
    predict_dataset = LMDataset(predict_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels, n=n)
    for i in range(predict_dataset.labels.shape[1]):
        logger.info(predict_dataset.label_types[i] + " - " + str(torch.unique(predict_dataset.labels[:, i])) + str(predict_dataset.label_classes))

    true_labels = predict_dataset.labels[:, 0]  

    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=3)

    model = ResNet.load_from_checkpoint(args.checkpoint)
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



            

def misclassify(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--prediction_file", type=str, required=True,
                        help="Location for file where all the true_labels and pred_labels have been stored")
    parser.add_argument("-o", "--output_dir", type=str, default="misclassified",
                        help="Directory to store the misclassified .npz files")
    
    args = parser.parse_args(argv)

    logger = get_logger('info')
    
    data = np.load(args.prediction_file)
    predictions = data['predictions']
    true_labels = data['true_labels']
    pred_labels = data['pred_labels']
    correct_incorrect = data['correct_incorrect']
    images = data['images'] 
    time_classes = data['time_classes']
    time_labels = data['time_labels']
    
    os.makedirs(args.output_dir, exist_ok=True)

    class_mapping = {i: label for i, label in enumerate(time_labels)}
    
    mapping_file = os.path.join(args.output_dir, "class_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("Class Mapping:\n")
        for index, label in class_mapping.items():
            f.write(f"Index {index} -> Class '{label}'\n")
    
    logger.info(f"Class mapping saved to {mapping_file}")

    misclassified_indices = np.where(correct_incorrect == 0)[0]

    unique_misclassifications = set(zip(true_labels[misclassified_indices], 
                                        pred_labels[misclassified_indices]))
    
    for true_label, pred_label in unique_misclassifications:
        indices = np.where((true_labels == true_label) & 
                           (pred_labels == pred_label) & 
                           (correct_incorrect == 0))[0]
        
        if len(indices) > 0:
            npz_filename = os.path.join(args.output_dir, f"misclassified_{true_label}_as_{pred_label}.npz")
            
            np.savez(npz_filename,
                     images=images[indices],
                     true_labels=true_labels[indices],
                     pred_labels=pred_labels[indices],
                     predictions=predictions[indices],
                     time_labels=time_labels[indices])
            
            logger.info(f"Saved {len(indices)} misclassified samples of class {true_label} predicted as {pred_label} to {npz_filename}")
        else:
            logger.info(f"No misclassified samples of class {true_label} predicted as {pred_label}")

    
        # Class Mapping:
        # Index 0 -> Class 'S10'
        # Index 1 -> Class 'S14'
        # Index 2 -> Class 'S4'


if __name__ == '__main__':
    print("Script is running directly.")    
    tune_train(num_samples=10, max_num_epochs=10, gpus_per_trial=1)
