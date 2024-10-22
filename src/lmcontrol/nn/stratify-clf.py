


####NOTE####
### This was solely done for the purpose of learning about the effects of "Batch Effects". It is very clear that batch effects is affecting the results. WE WONT BE USING THIS CODE ANYMORE###

###Note that here n means the total number of data points that shall be taken from each of the class

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
from ..utils import get_logger
from .dataset import get_transforms as _get_transforms
import wandb
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from lightning.pytorch.callbacks import EarlyStopping
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from ..data_utils import encode_labels
from ..utils import get_logger

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

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


#############################################################################################################################



class LMDataset(Dataset):
    def __init__(self, npzs, use_masks=False, return_labels=False, logger=None, transform=None, label_types=None, n=None,
                 train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15, split='train'):
        """
        Args:
            npzs (array-like)       : A list or tuple of paths to NPZ files containing cropped images
            split (str)             : Specifies which split of data to return ('train', 'val', 'test')
        """
        if not isinstance(npzs, (list, tuple, np.ndarray, torch.Tensor)):
            raise ValueError(f"Got unexpected type ({type(npzs)}) for argument 'npzs'. Must be an array-like")
        elif len(npzs) == 0:
            raise ValueError("Got empty array-like for argument 'npzs'")

        logger = logger or get_logger('warning')
        masks, images, paths, metadata = load_npzs(npzs, logger, n, label_types)
        
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

        if return_labels and split in ['train', 'val', 'test']:
            self._split_data(train_ratio, val_ratio, test_ratio, split)

    def _split_data(self, train_ratio, val_ratio, test_ratio, split):
        """
        Splits the dataset into training, validation, and testing sets.
        """
        #assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

        num_samples = len(self.data)
        indices = np.arange(num_samples)
        
        train_indices, temp_indices = train_test_split(indices, train_size=train_ratio, stratify=self.labels)
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(temp_indices, train_size=val_size, stratify=self.labels[temp_indices])

        if split == 'train':
            self.data = self.data[train_indices]
            if self.labels is not None:
                self.labels = self.labels[train_indices]
        elif split == 'val':
            self.data = self.data[val_indices]
            if self.labels is not None:
                self.labels = self.labels[val_indices]
        elif split == 'test':
            self.data = self.data[test_indices]
            if self.labels is not None:
                self.labels = self.labels[test_indices]

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

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        n_features = 512 
        self.backbone.fc = nn.Linear(n_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters() #this is an important part, we didnt add it before. STUDY WHY THIS MATTERED 

    def forward(self, x):
        return self.backbone(x) 

    def training_step(self, batch, batch_idx): 
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels[:, 0])
        
        # Logging accuracy
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):  #Added this part, read the pytorch documentation to understand how its used 
        x = batch[0]
        return self.forward(x).flatten(start_dim=1)    


def get_transform():
    transform = _get_transforms('float', 'norm', 'rgb')
    return transform



def stratify(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, help="the label to predict with")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("--S10", type=str, nargs='+', required=True, help="Provide files for S10")
    parser.add_argument("--S14", type=str, nargs='+', required=True, help="Provide files for S14")
    parser.add_argument("--S4", type=str, nargs='+', required=True, help="Provide files for S4")
    parser.add_argument("-tr", "--train_size", type=float, help="Training percent", default=0.6)
    parser.add_argument("-val", "--validate_size", type=float, help="Validation percent", default=0.2)
    parser.add_argument("-te", "--test_size", type=float, help="Testing percent", default=0.2)
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to the model checkpoint file to use for inference")
    parser.add_argument("-d", "--debug", action='store_true', help="Run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="The directory to save output to", default='.')
    parser.add_argument("-oe","--output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-n", "--data_size", type=int, help="Number of samples to use from each class (S10, S14, S4)", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="Enable early stopping", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")

    args = parser.parse_args(argv)
    
    logger = get_logger('info')
    transform = get_transform()
    
    n = args.data_size

    logger.info(f"loading datasets")
    S10_files = args.S10
    S14_files = args.S14
    S4_files = args.S4

    logger.info(f"loading datasets into LMDataset")
    train_dataset = LMDataset(S10_files + S14_files + S4_files, return_labels=True, transform=transform, split='train',logger=logger, label_types=args.labels,n=n)
    val_dataset = LMDataset(S10_files + S14_files + S4_files, return_labels=True, transform=transform, split='val',logger=logger, label_types=args.labels,n=n)
    test_dataset = LMDataset(S10_files + S14_files + S4_files, return_labels=True, transform=transform, split='test',logger=logger, label_types=args.labels,n=n)

    num_workers = 0 if args.debug else 4


    logger.info(f"loading datasets into DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)

    logger.info(f"Starting training")
    model = ResNet(num_classes=3)   # this is a quick fix, it will not work if we specify more than one label (Andrew)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    wandb.init(project="SX_HTY_Run1")
    wandb_logger = WandbLogger(project='your_project_name', log_model=True)

    callbacks = []
    if args.early_stopping:
        early_stopping = EarlyStopping(
         monitor="val_accuracy",  
         min_delta=0.000001,      
         patience=10,         
         verbose=False,      
         mode="max"          
        )
        callbacks.append(early_stopping)


    trainer = L.Trainer(
        max_epochs=args.epochs,                      
        devices=1,
        accelerator=accelerator,
        callbacks=callbacks,  
        logger=wandb_logger  
    )
    
    trainer.fit(model, train_loader, val_loader)


    if args.checkpoint:
        trainer.save_checkpoint(args.checkpoint)
        logger.info(f"Model saved to {args.checkpoint}")

    logger.info(f'Done with training, lets predict now')

    true_labels = test_dataset.labels[:, 0]  ##


    model = ResNet.load_from_checkpoint(args.checkpoint)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)

    logger.info("Running predictions")
    predictions = trainer.predict(model, test_loader)
    predictions = torch.cat(predictions).numpy()

    pred_labels = np.argmax(predictions, axis=1)  ##

    if true_labels is not None: ##
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
    correct_incorrect = np.where(pred_labels == true_labels ,1 ,0)  #maybe just remove this variable declared 
    out_data = dict(predictions = predictions, true_labels = true_labels, pred_labels = pred_labels, correct_incorrect = correct_incorrect)

    if not args.pred_only:
        dset = test_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        for i, k in enumerate(dset.label_types):
            out_data[k + "_classes"] = dset.label_classes[i]
            out_data[k + "_labels"] = np.asarray(dset.labels[:, i])

    
    np.savez(args.output_npz, **out_data)

if __name__ == '__main__':
    stratify()