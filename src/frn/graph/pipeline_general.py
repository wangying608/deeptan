r"""
The pipeline for training the model.
"""
import os
# from typing import Any, List, Dict, Optional, Union
import torch
import torch.nn as nn
import lightning as ltn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MatthewsCorrCoef
from torchmetrics.regression import MeanAbsoluteError, R2Score, PearsonCorrCoef
import frn.constants as MC
# Choose a version:
from frn.graph.core_gat_pyg import Backbone
# from frn.graph.core_pgat import Backbone


class MyGAT(ltn.LightningModule):
    def __init__(
            self,
            in_channels: int,
            graph_label_dim: int,
            regression: bool,
            # edge_dim: int,
            hidden_dim: int = MC.default.node_feature_dim,
            heads: int = MC.default.n_heads,
            lr: float = MC.default.lr,
            dropout: float = MC.default.dropout,
            negative_slope: float = MC.default.negative_slope,
        ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.is_regression = regression
        self.output_dim = graph_label_dim
        self.backbone_output_dim = graph_label_dim * 8
        self.hidden_dim = hidden_dim

        # self.model = Backbone(in_channels, self.backbone_output_dim, edge_dim, heads, dropout, negative_slope)
        self.model = Backbone(in_channels, self.backbone_output_dim, hidden_dim, heads, dropout, negative_slope)
        
        # Define the output layer for graph label prediction
        self.fc_1 = nn.Linear(self.backbone_output_dim, graph_label_dim)
        
        # Define the metrics
        self._define_metrics(graph_label_dim, self.is_regression)

    def forward(self, data_batch):
        x = self.model(data_batch)
        x = self.fc_1(x)
        # x = x.squeeze(-1)
        # if self.output_dim > 1 and not self.regression:
        #     x = nn.functional.log_softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        # x, edge_index, edge_attr, y = batch.x, batch.edge_index, batch.edge_attr, batch.y
        # y_hat = self(x, edge_index, edge_attr)
        y_pred = self.forward(batch)
        loss = self._my_loss(MC.title_train, batch.y, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        loss = self._my_loss(MC.title_val, batch.y, y_pred)
        return loss
    
    def test_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        loss = self._my_loss(MC.title_test, batch.y, y_pred)
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
                self.loss_fn = nn.MSELoss(reduction='none')
                # self.mae = MeanAbsoluteError()
                # self.r2 = R2Score()
                # self.pcc = PearsonCorrCoef()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
                self.mcc = MatthewsCorrCoef(task='multiclass', num_classes=output_dim)
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
    
    def _my_loss(self, which_step: str, y: torch.Tensor, y_pred: torch.Tensor):
        #!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.output_dim > 1 and not self.is_regression:
            y = y.argmax(dim=-1)

        if self.output_dim == 1:
            loss = self.loss_fn(y_pred, y)
            self.log(f"{which_step}_loss", loss, sync_dist=True)
            self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
            self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
        else:
            if self.is_regression:
                loss = self.loss_fn(y_pred, y)
                loss = loss.mean(dim=0)
                loss = loss.sum()
                self.log(f"{which_step}_loss", loss, sync_dist=True)
                # self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
                # if y.shape[0] < 2:
                #     return loss
                # self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
                # self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
            else:
                loss = self.loss_fn(y_pred, y)
                self.log(f"{which_step}_loss", loss, sync_dist=True)
                
                self.log(f"{which_step}_mcc", self.mcc(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_f1_micro", self.f1_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_f1_macro", self.f1_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_f1_weighted", self.f1_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_recall_micro", self.recall_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_recall_macro", self.recall_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_recall_weighted", self.recall_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_precision_micro", self.precision_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_precision_macro", self.precision_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_precision_weighted", self.precision_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_accuracy_micro", self.accuracy_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_accuracy_macro", self.accuracy_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_accuracy_weighted", self.accuracy_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_auroc_macro", self.auroc_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_auroc_weighted", self.auroc_weighted(y_pred, y), sync_dist=True)
        
        return loss

