import argparse
import glob
import lightning as L
import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader
import glob
import numpy as np
import argparse
import lightning as L
from ..utils import get_logger
from ..utils import get_logger
from .dataset import LMDataset, get_transforms as _get_transforms
import wandb
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from lightning.pytorch.callbacks import EarlyStopping
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping



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

    # def training_step(self, batch, batch_idx): 
    #     images, labels = batch
    #     outputs = self.forward(images)
    #     loss = self.criterion(outputs, labels[:, 0])
    #     self.log('train_loss', loss, batch_size=images.size(0))
    #     return loss
      

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

    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch
    #     outputs = self.forward(images)
    #     loss = self.criterion(outputs, labels[:, 0])
    #     self.log('val_loss', loss, batch_size=images.size(0))
    #     return loss 

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



def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument('labels', type=str, nargs='+', help="the label to train with")
    parser.add_argument("--training", type=str, nargs='+', required=True, help="directories containing training data")
    parser.add_argument("--validation", type=str, nargs='+', required=True, help="directories containing validation data")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")    
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')

    args = parser.parse_args(argv)

    logger = get_logger('info')

    transform = get_transform()  

    train_files = args.training
    val_files = args.validation
    

    logger.info(f"Loading training data: {len(train_files)} files")
    train_dataset = LMDataset(train_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels)
    for i in range(train_dataset.labels.shape[1]):
        logger.info(train_dataset.label_types[i] + " - " + str(torch.unique(train_dataset.labels[:, i])) + str(train_dataset.label_classes))
    


    
    logger.info(f"Loading validation data: {len(val_files)} files")
    val_dataset = LMDataset(val_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels)
    for i in range(val_dataset.labels.shape[1]):
        logger.info(val_dataset.label_types[i] + " - " + str(torch.unique(val_dataset.labels[:, i])) + str(val_dataset.label_classes))

    num_workers = 0 if args.debug else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)

    model = ResNet(num_classes=len(train_dataset.label_classes[0]))   # this is a quick fix, it will not work if we specify more than one label
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    wandb.init(project="SX_HTY_Run1")
    wandb_logger = WandbLogger(project='your_project_name', log_model=True)
    # trainer = L.Trainer(max_epochs=args.epochs, devices=1, logger=wandb_logger, accelerator=accelerator)
    # trainer.fit(model, train_loader, val_loader)

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_accuracy",  
        min_delta=0.000001,      
        patience=10,         
        verbose=False,      
        mode="max"          
    )


    trainer = L.Trainer(
        max_epochs=args.epochs,                       ##CHANGE THIS TO 100
        devices=1,
        accelerator=accelerator,
        callbacks=[early_stopping],  #UNCOMMENT WHEN DONE
        logger=wandb_logger  
    )
    
    trainer.fit(model, train_loader, val_loader)


    if args.checkpoint:
        trainer.save_checkpoint(args.checkpoint)
        logger.info(f"Model saved to {args.checkpoint}")


def predict(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, nargs='+', help="the label to predict with")
    parser.add_argument("--prediction", type=str, nargs='+', required=True, help="directories containing prediction data")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")      
    parser.add_argument("-o","--output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")

    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = get_transform()  

    predict_files = args.prediction

    logger.info(f"Loading predicting data: {len(predict_files)} files")
    predict_dataset = LMDataset(predict_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels)

    for i in range(predict_dataset.labels.shape[1]):
        logger.info(predict_dataset.label_types[i] + " - " + str(torch.unique(predict_dataset.labels[:, i])) + str(predict_dataset.label_classes))

    true_labels = predict_dataset.labels[:, 0]  ##

    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=3)

    model = ResNet.load_from_checkpoint(args.checkpoint)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(devices=1, accelerator=accelerator)

    logger.info("Running predictions")
    predictions = trainer.predict(model, predict_loader)
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

    # out_data = dict(predictions=predictions)
    out_data = dict(predictions=predictions, true_labels=true_labels)  ##Recently added


    if not args.pred_only:
        dset = predict_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        for i, k in enumerate(dset.label_types):
            out_data[k + "_classes"] = dset.label_classes[i]
            out_data[k + "_labels"] = np.asarray(dset.labels[:, i])

    np.savez(args.output_npz, **out_data)

if __name__ == '__main__':
    train()
