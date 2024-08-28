r"""
The pipeline for training the model.
"""
import os
from typing import Any, List, Dict, Optional, Union
import torch
import torch.nn as nn
import lightning as ltn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, R2Score, PearsonCorrCoef
# from torch_geometric.data import Data as PyG_Data
import torch_geometric.loader as geom_loader
# from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import FakeDataset, TUDataset
from torch_geometric.data.lightning import LightningDataset as PyG_LightningDataset
from .core_gat_pyg import MyGATModel


class MyGAT(ltn.LightningModule):
    def __init__(
            self,
            in_channels: int,
            graph_label_dim: int,
            regression: bool,
            edge_dim: int,
            heads: int = 4,
            lr: float = 0.001,
            dropout: float = 0.6,
            negative_slope: float = 0.2,
        ):
        super().__init__()
        # self.save_hyperparameters()
        self.regression = regression
        self.lr = lr
        self.output_dim = graph_label_dim
        self.model = MyGATModel(in_channels, graph_label_dim, edge_dim, heads, dropout, negative_slope)
        self._define_metrics(graph_label_dim, self.regression)

    def forward(self, data_batch):
        x = self.model(data_batch)
        # x = x.squeeze(-1)
        return x
    
    def training_step(self, batch, batch_idx):
        # x, edge_index, edge_attr, y = batch.x, batch.edge_index, batch.edge_attr, batch.y
        # y_hat = self(x, edge_index, edge_attr)
        y_hat = self.forward(batch)
        loss = self._my_loss(y_hat, batch.y, "train", self.regression)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self._my_loss(y_hat, batch.y, "val", self.regression)
        return loss
    
    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self._my_loss(y_hat, batch.y, "test", self.regression)
        return loss
    
    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _define_metrics(self, output_dim: int, regression: bool):
        """
        Define the loss function and the metrics.
        """
        if output_dim == 1:
            self.loss_fn = nn.MSELoss()
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score()
            self.pcc = PearsonCorrCoef()
        else:
            if regression:
                # Multi-label regression
                self.loss_fn = nn.MSELoss()
                self.mae = MeanAbsoluteError()
                self.r2 = R2Score()
                self.pcc = PearsonCorrCoef()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
                self.recall_micro = MulticlassRecall(average="micro", num_classes=output_dim)
                self.recall_macro = MulticlassRecall(average="macro", num_classes=output_dim)
                self.recall_weighted = MulticlassRecall(average="weighted", num_classes=output_dim)
                #
                self.precision_micro = MulticlassPrecision(average="micro", num_classes=output_dim)
                self.precision_macro = MulticlassPrecision(average="macro", num_classes=output_dim)
                self.precision_weighted = MulticlassPrecision(average="weighted", num_classes=output_dim)
                #
                self.f1_micro = MulticlassF1Score(average="micro", num_classes=output_dim)
                self.f1_macro = MulticlassF1Score(average="macro", num_classes=output_dim)
                self.f1_weighted = MulticlassF1Score(average="weighted", num_classes=output_dim)
                #
                self.accuracy_micro = MulticlassAccuracy(average="micro", num_classes=output_dim)
                self.accuracy_macro = MulticlassAccuracy(average="macro", num_classes=output_dim)
                self.accuracy_weighted = MulticlassAccuracy(average="weighted", num_classes=output_dim)
                #
                self.auroc_macro = MulticlassAUROC(average="macro", num_classes=output_dim)
                self.auroc_weighted = MulticlassAUROC(average="weighted", num_classes=output_dim)
    
    def _my_loss(self, y_hat: torch.Tensor, y: torch.Tensor, which_step: str, regression: bool):
        #!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.output_dim > 1 and not regression:
            y = y.argmax(dim=-1)
        
        loss = self.loss_fn(y_hat, y)
        self.log(f"{which_step}_loss", loss, sync_dist=True)

        if self.output_dim == 1:
            self.log(f"{which_step}_mae", self.mae(y_hat, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log(f"{which_step}_pcc", self.pcc(y_hat, y), sync_dist=True)
            self.log(f"{which_step}_r2", self.r2(y_hat, y), sync_dist=True)
        else:
            if regression:
                self.log(f"{which_step}_mae", self.mae(y_hat, y), sync_dist=True)
                if y.shape[0] < 2:
                    return loss
                self.log(f"{which_step}_pcc", self.pcc(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_r2", self.r2(y_hat, y), sync_dist=True)
            else:
                self.log(f"{which_step}_f1_micro", self.f1_micro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_f1_macro", self.f1_macro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_f1_weighted", self.f1_weighted(y_hat, y), sync_dist=True)
                #
                self.log(f"{which_step}_recall_micro", self.recall_micro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_recall_macro", self.recall_macro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_recall_weighted", self.recall_weighted(y_hat, y), sync_dist=True)
                #
                self.log(f"{which_step}_precision_micro", self.precision_micro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_precision_macro", self.precision_macro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_precision_weighted", self.precision_weighted(y_hat, y), sync_dist=True)
                #
                self.log(f"{which_step}_accuracy_micro", self.accuracy_micro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_accuracy_macro", self.accuracy_macro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_accuracy_weighted", self.accuracy_weighted(y_hat, y), sync_dist=True)
                #
                self.log(f"{which_step}_auroc_macro", self.auroc_macro(y_hat, y), sync_dist=True)
                self.log(f"{which_step}_auroc_weighted", self.auroc_weighted(y_hat, y), sync_dist=True)
        
        return loss


def train_graph(
        model,
        dataloader_train,
        dataloader_val,
        es_patience: int,
        max_epochs: int,
        min_epochs: int,
        log_dir: str,
        devices: Union[list[int], str, int] = 'auto',
        accelerator: str = 'auto',
        in_dev: bool = False,
    ):
    """
    Fit the graph model on a PyTorch Geometric dataset.
    """
    if type(devices) == int and device_count() > 0:
        avail_dev = find_usable_cuda_devices(devices)
    elif devices == 'auto' and device_count() > 0:
        avail_dev = find_usable_cuda_devices()
    else:
        avail_dev = devices

    callback_es = EarlyStopping(
        monitor='val_loss',
        patience=es_patience,
        mode='min',
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename='best-model-{epoch:03d}-{val_loss:.3f}',
        monitor='val_loss',
    )

    logger_tr = TensorBoardLogger(
        save_dir=log_dir,
        name='',
    )

    trainer = ltn.Trainer(
        fast_dev_run=in_dev,
        logger=logger_tr,
        log_every_n_steps=1,
        # precision='16-mixed',
        devices=avail_dev,
        accelerator=accelerator,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[callback_es, callback_ckpt],
        num_sanity_val_steps=0,
        default_root_dir=log_dir,
    )
    
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    return callback_ckpt.best_model_score.item()


# n_channels=64
# edge_dim=16
# graph_label_dim=1

# datamodule = PyG_LightningDataset(
#     train_dataset=FakeDataset(num_graphs=300, num_channels=n_channels, edge_dim=edge_dim, task="graph"),
#     val_dataset=FakeDataset(num_graphs=100, num_channels=n_channels, edge_dim=edge_dim, task="graph"),
#     # test_dataset=FakeDataset(num_graphs=100, num_channels=64, edge_dim=16, task="graph"),
# )
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_datasets")
BATCH_SIZE = 16

if __name__ == "__main__":
    tu_dataset = TUDataset(root=DATASET_PATH, name="MUTAG")
    torch.manual_seed(42)
    tu_dataset.shuffle()
    train_dataset = tu_dataset[:200]
    val_dataset = tu_dataset[200:]
    graph_train_loader = geom_loader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    graph_val_loader = geom_loader.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    dim_node_feat = tu_dataset.num_node_features
    n_graph_label_class = tu_dataset.num_classes
    dim_edge_feat = tu_dataset.num_edge_features
    print("\n", dim_node_feat, n_graph_label_class, dim_edge_feat, "\n")

    loss_min = train_graph(
        model=MyGAT(
            in_channels=dim_node_feat,
            graph_label_dim=n_graph_label_class,
            regression=False,
            edge_dim=dim_edge_feat,
            heads=4,
            lr=0.001,
            dropout=0.2,
            negative_slope=0.2,
        ),
        dataloader_train=graph_train_loader,
        dataloader_val=graph_val_loader,
        es_patience=15,
        max_epochs=1000,
        min_epochs=10,
        log_dir="/home/wuch/Downloads/.tmp/runs/00",
    )
