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

from sklearn.preprocessing import LabelEncoder
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet18

class MyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)
        image = data['image']
        label = data['label']

        if self.transform:
            image = self.transform(image)

        return image, label
    
class ResNetModel(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_features = 64

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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

def get_transform(transform1=None):
    if transform1 is None:
        transform1 = _get_transforms('float')
    return transform1

def get_npzs(timepoints, hts):
    ret = list()
    for tp in timepoints:
        for ht in hts:
            ret.extend(glob.glob(f"S{tp}/*HT{ht}/*.npz"))
    return ret



def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument("--training", type=str, help="directory containing training data")
    parser.add_argument("--validation", type=str, help="directory containing validation data")
    parser.add_argument("--testing", type=str, help="directory containing testing data")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)

    args = parser.parse_args(argv)


    # Load datasets
    train_dataset = MyDataset(args.training, transform=transforms.ToTensor())
    val_dataset = MyDataset(args.validation, transform=transforms.ToTensor())
    test_dataset = MyDataset(args.testing, transform=transforms.ToTensor())

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Define model
    model = ResNetModel()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(args.epochs):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {100. * correct / len(val_loader.dataset)}%')

    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset)}%')

if __name__ == '__main__':
    train()

    Xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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

    model = ResNetModel()

    trainer = L.Trainer(max_epochs=args.epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    # After training, call the test function
    test(trainer, model, test_loader)

def test(trainer, model, test_loader):
    trainer.test(model, test_loader)

if __name__ == '__main__':
    train()
