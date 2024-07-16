
import argparse
import copy
import glob

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset
import torchvision
from torch import nn

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import BYOLTransform
from lightly.utils.scheduler import cosine_schedule

from ..utils import get_logger
from .dataset import get_lightly_dataset, get_transforms as _get_transforms


class ResNet(L.LightningModule):

    # Note: The model and training settings do not follow the reference settings
    # from the paper. The settings are chosen such that the example can easily be
    # run on a small dataset with a single GPU.

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, num_classes=10):
        super().__init__()
        # Load the pretrained ResNet18 model
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_features = 512  # Number of features for ResNet18
        self.backbone.fc = nn.Linear(n_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)
    

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, batch_size=images.size(0))
        return loss


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)
    
    #add scheduler too

def get_transform(transform1=None):
    """Get BYOL transform

    By default, the two transforms are:
        transform1: rotate, crop, hflip, vflip, float, rgb
        transform2: blur, rotate, crop, hflip, vflip, float, noise, rgb

    For more details on these transforms, see lmcontrol.nn.dataset.get_transform
    """
    if transform1 is None:
        transform1 = _get_transforms('float', 'norm', 'rotate', 'crop', 'hflip', 'vflip', 'rgb')
  

    transform = BYOLTransform(
        view_1_transform=transform1,
        
    )
    return transform


def get_npzs(timepoints, hts):
    ret = list()
    for tp in timepoints:
        for ht in hts:
            ret.extend(glob.glob(f"S{tp}/*HT{ht}/*.npz"))
    return ret

class MyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.file_list[idx]
        label = self._get_label(img_name)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label(self, file_path): #finds the label associated 

        label = file_path.split('_')[-2]  
        return label


def get_npzs(timepoints, hts):
    ret = []
    for tp in timepoints:
        for ht in hts:
            ret.extend(glob.glob(f"S{tp}_H{ht}_*/**/*.npz", recursive=True))
    return ret

def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument("--training", type=str, required=True, help="directory containing training data")
    parser.add_argument("--validation", type=str, required=True, help="directory containing validation data")
    parser.add_argument("--testing", type=str, required=True, help="directory containing testing data")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)

    args = parser.parse_args(argv)

    logger = CSVLogger("logs", name=args.experiment)

    transform = get_transform()

    train_files = get_npzs(args.training)
    val_files = get_npzs(args.validation)
    test_files = get_npzs(args.testing)

    train_dataset = MyDataset(train_files, transform=transform)
    val_dataset = MyDataset(val_files, transform=transform)
    test_dataset = MyDataset(test_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ResNet()

    trainer = L.Trainer(max_epochs=args.epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    # After training, call the test function
    test(trainer, model, test_loader)

def test(trainer, model, test_loader):
    trainer.test(model, test_loader)

if __name__ == '__main__':
    train()
