import copy

import lightning as L
import torch
import torchvision
from torch import nn

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import BYOLTransform
from lightly.utils.scheduler import cosine_schedule

from .dataset import get_transforms as _get_transforms


class BYOL(L.LightningModule):

    # Note: The model and training settings do not follow the reference settings
    # from the paper. The settings are chosen such that the example can easily be
    # run on a small dataset with a single GPU.

    val_metric = "validation_ncs"
    train_metric = "train_ncs"

    def __init__(self, model="resnet18"):
        super().__init__()
        if model == "resnet18":
            resnet = torchvision.models.resnet18()
            n_features = 512
        elif model == "resnet50":
            resnet = torchvision.models.resnet50()
            n_features = 2048
        elif model == "convnext_tiny":
            resnet = torchvision.models.convnext_tiny()
            n_features = 768
        else:
            raise ValueError(f"Unrecognized model: '{model}'")


        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(n_features, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1) = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.train_metric, loss, batch_size=x0.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1) = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(self.val_metric, loss, batch_size=x0.size(0))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        return self.backbone(x).flatten(start_dim=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)


def get_transform(transform1=None, transform2=None):
    """Get BYOL transform

    By default, the two transforms are:
        transform1: rotate, crop, hflip, vflip, float, rgb
        transform2: blur, rotate, crop, hflip, vflip, float, noise, rgb

    For more details on these transforms, see lmcontrol.nn.dataset.get_transform
    """
    if transform1 is None:
        transform1 = _get_transforms('float', 'norm', 'rotate', 'crop', 'hflip', 'vflip', 'rgb')
    if transform2 is None:
        transform2 = _get_transforms('float', 'norm', 'blur', 'rotate', 'crop', 'hflip', 'vflip', 'noise', 'rgb')

    transform = BYOLTransform(
        view_1_transform=transform1,
        view_2_transform=transform2
    )
    return transform
