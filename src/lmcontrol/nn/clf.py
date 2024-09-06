
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

from optuna.integration import PyTorchLightningPruningCallback
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


def _add_training_args(parser):
    parser.add_argument('labels', type=str, help="the label to train with") #Let us currently only work with single label. We will add other labels in future (Andrew)
    parser.add_argument("--training", type=str, nargs='+', required=True, help="directories containing training data")
    parser.add_argument("--validation", type=str, nargs='+', required=True, help="directories containing validation data")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-e", "--epochs", type=int, help="the number of epochs to run for", default=10)
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-n", "--data_size", type=int, help="number of samples to use from each class", default=None)
    parser.add_argument("--early_stopping", action='store_true', help="enable early stopping", default=False)


def _get_loaders_and_model(args):
    transform = get_transform()

    train_files = args.training
    val_files = args.validation

    n = args.data_size

    logger.info(f"Loading training data: {len(train_files)} files")
    train_dataset = LMDataset(train_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)
    for i in range(train_dataset.labels.shape[1]):
        logger.info(train_dataset.label_types[i] + " - " + str(torch.unique(train_dataset.labels[:, i])) + str(train_dataset.label_classes))


    logger.info(f"Loading validation data: {len(val_files)} files")
    val_dataset = LMDataset(val_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)
    for i in range(val_dataset.labels.shape[1]):
        logger.info(val_dataset.label_types[i] + " - " + str(torch.unique(val_dataset.labels[:, i])) + str(val_dataset.label_classes))

    num_workers = 0 if args.debug else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)

    model = ResNet(num_classes=len(train_dataset.label_classes[0]),
                   lr=args.lr, args.step_size=step_size, gamma=args.gamma)

    return train_loader, val_loader, model


def _get_trainer(args, hpo=False):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    targs = dict(max_epochs=args.epochs, devices=1, accelerator=accellerator, callbacks=callbacks)

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


    if hpo:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_accuracy"))
        targs['logger'] = CSVLogger(args.outdir, name=args.experiment)
    else:
        wandb.init(project="SX_HTY_Run1")
        targs['logger'] = WandbLogger(project='your_project_name', log_model=True)

    return L.Trainer(**targs)


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument('labels', type=str, help="the label to train with") #Let us currently only work with single label. We will add other labels in future (Andrew)

    _add_training_args(parser)

    args = parser.parse_args(argv)

    logger = get_logger('info')

#     transform = get_transform()
#
#     train_files = args.training
#     val_files = args.validation
#
#     n = args.data_size
#
#     logger.info(f"Loading training data: {len(train_files)} files")
#     train_dataset = LMDataset(train_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)
#     for i in range(train_dataset.labels.shape[1]):
#         logger.info(train_dataset.label_types[i] + " - " + str(torch.unique(train_dataset.labels[:, i])) + str(train_dataset.label_classes))
#
#
#     logger.info(f"Loading validation data: {len(val_files)} files")
#     val_dataset = LMDataset(val_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)
#     for i in range(val_dataset.labels.shape[1]):
#         logger.info(val_dataset.label_types[i] + " - " + str(torch.unique(val_dataset.labels[:, i])) + str(val_dataset.label_classes))
#
#     num_workers = 0 if args.debug else 4
#
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=num_workers)
#
#     model = ResNet(num_classes=len(train_dataset.label_classes[0]))   # this is a quick fix, it will not work if we specify more than one label
#     accelerator = "gpu" if torch.cuda.is_available() else "cpu"
#
#     trainer = _get_trainer(args)

    train_loader, val_loader, model = _get_loaders_and_model(args)
    trainer = _get_trainer(args)
    trainer.fit(model, train_loader, val_loader)

    trainer.fit(model, train_loader, val_loader)


def objective(args, trial):

    args.batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.step_size = trial.suggest_int('step_size', 5, 15, step=5)
    args.gamma = trial.suggest_float('gamma', 0.1, 0.5)
    args.outdir = os.path.join(args.working_dir, "logs")
    args.experiment = f"trial_{trial.number:04d}"

    train_loader, val_loader, model = _get_loaders_and_model(args)
    trainer = _get_trainer(args)
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_accuracy"].item()


def tune(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", type=str, help="the SQLite database to use")
    parser.add_argument("-t", "--n_trials", type=int, help="the number of trials to run", default=1)

    _add_training_args(parser)

    args = parser.parse_args(argv)

    pkl = os.path.join(args.working_dir, "args.pkl")
    if os.path.exists(pkl):
        with open(pkl, 'rb') as file:
            args = pickle.load(file)
    else:
        os.path.mkdirs(args.working_dir)
        with open(pkl, 'wb') as file:
            pickle.dump(args, file)

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    #study = optuna.create_study(direction="maximize", pruner=pruner)

    obj = partial(objective, args)

    study = optuna.create_study(study="study", storage=f"sqlite:///{args.working_dir}/study.db", load_if_exists=True, 'maximize')

    study.optimize(obj, n_trials=args.n_trials)


def predict(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, help="the label to predict with") #Let us currently only work with single label. We will add other labels in future  (Andrew)
    parser.add_argument("--prediction", type=str, nargs='+', required=True, help="directories containing prediction data")
    parser.add_argument("-c","--checkpoint", type=str, help="path to the model checkpoint file to use for inference")
    parser.add_argument("-o","--output_npz", type=str, help="the path to save the embeddings to. Saved in NPZ format")
    parser.add_argument("-d", "--debug", action='store_true', help="run with a small dataset", default=False)
    parser.add_argument("-p", "--pred-only", action='store_true', default=False, help="only save predictions, otherwise save original image data and labels in output_npz")
    parser.add_argument("-n", "--data_size", type=int, help="number of samples to use from each class", default=None)


    args = parser.parse_args(argv)

    logger = get_logger('info')
    transform = get_transform()

    predict_files = args.prediction
    n = args.data_size


    logger.info(f"Loading predicting data: {len(predict_files)} files")
    predict_dataset = LMDataset(predict_files, transform=transform, logger=logger, return_labels=True, label_types=args.labels,n=n)
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

    #NEW STUFF FOR 02/08
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    correct_incorrect = np.where(pred_labels == true_labels ,1 ,0)  #maybe just remove this variable declared
    out_data = dict(predictions = predictions, true_labels = true_labels, pred_labels = pred_labels, correct_incorrect = correct_incorrect)

    if not args.pred_only:
        dset = predict_dataset
        out_data['images'] = np.asarray(torch.squeeze(dset.data))
        for i, k in enumerate(dset.label_types):
            out_data[k + "_classes"] = dset.label_classes[i]
            out_data[k + "_labels"] = np.asarray(dset.labels[:, i])

    #np.savez(args.output_npz, **out_data)
    #NEW STUFF
    # np.savez(args.output_npz, predictions = predictions, true_labels = true_labels, pred_labels = pred_labels, correct_incorrect = correct_incorrect)
    np.savez(args.output_npz, **out_data)


if __name__ == '__main__':
    train()
