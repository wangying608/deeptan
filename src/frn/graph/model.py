r"""
MSGP for semi-supervised multi-task learning.
"""
from typing import List
import torch
from torch.optim.adam import Adam
import lightning as ltn
from torch_geometric.data import Data as GData
# from torchmetrics.wrappers import MultitaskWrapper#, MultioutputWrapper
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MatthewsCorrCoef
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
import frn.constants as const
from frn.graph.core import MSGP
from frn.graph.modules import VGAE_Decoder, GLabelPredictor, AttPool

torch.set_float32_matmul_precision(const.default.matmul_precision)


class MSGPMTL(ltn.LightningModule):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            is_regression: bool,
            output_dims_nd: List[int],
            output_dim_g_emb: int,
            n_hop: int,
            threshold_subgraph_overlap: float,
            n_heads: List[int],
            dropout: float,
            lr: float,
            negative_slope: float,
        ):
        r"""Multi-task learning model.

        Args:
            input_dim: Input node embedding dimension.

            output_dim: Number of output classes. If it is 1, the model will be a regression model. Otherwise, it should be at least 3 (3 for binary classification) for classification tasks.
            
            is_regression: Whether the task is a regression task or not.

            output_dims_nd: Output node embedding dimensions for each layer.
                The length denotes the number of layers.
            
            output_dim_g_emb: Output graph embedding dimension.

            n_hop: Maximum number of hops for searching central nodes' neighbors.
                ``n_hop >= 2`` is necessary to graph attention.

            threshold_subgraph_overlap: Threshold for the overlap between subgraphs.

            n_heads: Number of attention heads.
            
            dropout: Dropout rate.
            
            lr: Learning rate for the optimizer.

            negative_slope: Negative slope for leaky ReLU.
        
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_regression = is_regression
        self.output_dims_nd = output_dims_nd
        self.output_dim_g_emb = output_dim_g_emb
        self.n_hop = n_hop
        self.threshold_subgraph_overlap = threshold_subgraph_overlap
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.negative_slope = negative_slope

        self.msgp = MSGP(input_dim, output_dims_nd, output_dim_g_emb, n_heads, n_hop, threshold_subgraph_overlap, negative_slope)
        self.ge_decoder = VGAE_Decoder(output_dim_g_emb, output_dims_nd[-1])
        self.label_predictor = GLabelPredictor(output_dim_g_emb, output_dim)
        self.pool_for_g_compare = AttPool(output_dims_nd[-1])

    def forward(self, g: GData):
        r"""
        Args:
            g: Graph data.
        
        Returns:
            x: Graph embedding.

            x_recon: Reconstructed node embeddings.

            x_label: Predicted graph label.

            g_: Graph data of aggregated and pooled ``g``. The size is same as ``g``.
            
            g_ms: Graph data that nodes are embedded multi-scale subgraphs. Its size is smaller than ``g``.
            
            x_recon_emb: Embedding of reconstructed node embeddings.
            
            g_emb: Embedding of graph ``g_`` nodes.
            
        """
        x, g_, g_ms = self.msgp(g)
        x_recon = self.ge_decoder(x)
        x_label = self.label_predictor(x)

        # For reconstruction loss
        x_recon_emb = self.pool_for_g_compare(x_recon)
        g_emb = self.pool_for_g_compare(g_.x)

        return x, x_recon, x_label, g_, g_ms, x_recon_emb, g_emb
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'monitor': const.dkey.title_val_loss,
                    }
                }
    
    def training_step(self, batch, batch_idx):
        prefix = const.dkey.abbr_train + '_'

        g = batch#[const.dkey.graph_]
        y = g.y
        x, x_recon, x_label, g_, g_ms, x_recon_emb, g_emb = self(g)

        loss_recon = self._loss_recon(prefix=prefix, x_recon_emb=x_recon_emb, g_emb=g_emb)

        if y is None:
            loss = loss_recon
        else:
            loss_pred = self._loss_pred(prefix=prefix, y=y, y_pred=x_label)
            loss = (loss_pred + loss_recon) / 2.0

        self.log(f"{prefix}loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        prefix = const.dkey.abbr_val + '_'
        g = batch#[const.dkey.graph_]
        y = g.y
        x, x_recon, x_label, g_, g_ms, x_recon_emb, g_emb = self(g)
        loss_recon = self._loss_recon(prefix=prefix, x_recon_emb=x_recon_emb, g_emb=g_emb)
        if y is None:
            loss = loss_recon
        else:
            loss_pred = self._loss_pred(prefix=prefix, y=y, y_pred=x_label)
            loss = (loss_pred + loss_recon) / 2.0
        self.log(f"{prefix}loss", loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        prefix = const.dkey.abbr_test + '_'
        g = batch#[const.dkey.graph_]
        y = g.y
        x, x_recon, x_label, g_, g_ms, x_recon_emb, g_emb = self(g)
        loss_recon = self._loss_recon(prefix=prefix, x_recon_emb=x_recon_emb, g_emb=g_emb)
        if y is None:
            loss = loss_recon
        else:
            loss_pred = self._loss_pred(prefix=prefix, y=y, y_pred=x_label)
            loss = (loss_pred + loss_recon) / 2.0
        self.log(f"{prefix}loss", loss, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        g = batch#[const.dkey.graph_]
        return self(g)
    
    def _loss_recon(self, prefix: str, x_recon_emb: torch.Tensor, g_emb: torch.Tensor):
        r"""Compute the reconstruction loss for the graph embedding based on the structural similarity.
        """
        loss = torch.nn.functional.mse_loss(x_recon_emb, g_emb)
        self.log(f"{prefix}loss_recon", loss, sync_dist=True)
        return loss
    
    def _loss_pred(self, prefix: str, y: torch.Tensor, y_pred: torch.Tensor):
        self._def_metrics_label_pred(prefix)
        metrics_label = self.metrics_label_pred(y_pred, y)

        if self.is_regression:
            loss = metrics_label[f"{prefix}MSE"].mean()
            self.log(f"{prefix}loss_pred", loss, sync_dist=True)
            self.log_dict(metrics_label, sync_dist=True)
        else:
            loss = metrics_label[f"{prefix}F1_weighted"].mean().neg().add(1.0)
            self.log(f"{prefix}loss_pred", loss, sync_dist=True)
            self.log_dict(metrics_label, sync_dist=True)

        return loss

    def _def_metrics_label_pred(self, prefix: str):
        r"""Define the loss function and the metrics.
        """
        output_dim = self.output_dim
        if output_dim == 1:
            # The task is regression
            # self.metrics_label_pred = MetricCollection({"MSE": MeanSquaredError(), "MAE": MeanAbsoluteError(), "R2": R2Score(), "PCC": PearsonCorrCoef()}, prefix=prefix)
            self.metrics_label_pred = MetricCollection({"MSE": MeanSquaredError(), "MAE": MeanAbsoluteError()}, prefix=prefix)
        else:
            if self.is_regression:
                # The task is multi-trait regression. Compute loss and metrics per trait instead of average over all traits.
                # self.metrics_label_pred = MetricCollection({"MSE": MeanSquaredError(num_outputs=output_dim), "MAE": MeanAbsoluteError(num_outputs=output_dim), "R2": R2Score(num_outputs=output_dim), "PCC": PearsonCorrCoef(num_outputs=output_dim)}, prefix=prefix)
                self.metrics_label_pred = MetricCollection({"MSE": MeanSquaredError(num_outputs=output_dim), "MAE": MeanAbsoluteError(num_outputs=output_dim)}, prefix=prefix)
            else:
                # The task is classification
                self.metrics_label_pred = MetricCollection({
                    "F1_weighted": MulticlassF1Score(average="weighted", num_classes=output_dim),
                    "F1_micro": MulticlassF1Score(average="micro", num_classes=output_dim),
                    "F1_macro": MulticlassF1Score(average="macro", num_classes=output_dim),
                    "AUROC_weighted": MulticlassAUROC(average="weighted", num_classes=output_dim),
                    "AUPRC_macro": MulticlassAUROC(average="macro", num_classes=output_dim),
                    "Accuracy_weighted": MulticlassAccuracy(average="weighted", num_classes=output_dim),
                    "Accuracy_micro": MulticlassAccuracy(average="micro", num_classes=output_dim),
                    "Accuracy_macro": MulticlassAccuracy(average="macro", num_classes=output_dim),
                    "Precision_weighted": MulticlassPrecision(average="weighted", num_classes=output_dim),
                    "Precision_micro": MulticlassPrecision(average="micro", num_classes=output_dim),
                    "Precision_macro": MulticlassPrecision(average="macro", num_classes=output_dim),
                    "Recall_weighted": MulticlassRecall(average="weighted", num_classes=output_dim),
                    "Recall_micro": MulticlassRecall(average="micro", num_classes=output_dim),
                    "Recall_macro": MulticlassRecall(average="macro", num_classes=output_dim),
                    "MCC": MatthewsCorrCoef(task='multiclass', num_classes=output_dim),
                    "MSE": MeanSquaredError(num_outputs=output_dim),
                }, prefix=prefix)
        
        self.metrics_label_pred.to(self.device)
