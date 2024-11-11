r"""
MSGP for semi-supervised multi-task learning.
"""
from typing import List
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import lightning as ltn
from torch_geometric.data import Data as GData
from torchmetrics.wrappers import MultitaskWrapper, MultioutputWrapper
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MatthewsCorrCoef
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
import frn.constants as const
from frn.graph.core import MSGP

torch.set_float32_matmul_precision(const.default.matmul_precision)


class VGAE_Decoder(nn.Module):
    def __init__(self, graph_embedding_dim: int, node_embedding_dim: int):
        super().__init__()
        self.graph_embedding_dim = graph_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        
        self.guess_num_nodes = nn.Sequential(nn.Linear(graph_embedding_dim, 1))
        self.decoder = nn.Linear(graph_embedding_dim, node_embedding_dim)
    
    def forward(self, graph_embedding: torch.Tensor):
        num_nodes = self.guess_num_nodes(graph_embedding).detach().round().long()
        graph_embedding_expanded = graph_embedding.unsqueeze(0).expand(num_nodes, -1)
        # Shape of graph_embedding_expanded: (num_nodes, graph_embedding_dim)
        node_embeddings = self.decoder(graph_embedding_expanded)
        
        return node_embeddings
    

class GLabelPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MSGPSSL(ltn.LightningModule):
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
        r"""MSGP for supervised learning.

        Args:
            input_dim: Input node embedding dimension.

            output_dim: Number of output classes. If it is 1, the model will be a regression model. Otherwise, it should be at least 3 (3 for binary classification) for classification tasks.
            
            is_regression: Whether the task is a regression task or not.
            
            dropout: Dropout rate.
            
            lr: Learning rate for the optimizer.
        
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

        self._define_metrics(output_dim, is_regression)

        self.msgp = MSGP(input_dim, output_dims_nd, output_dim_g_emb, n_heads, n_hop, threshold_subgraph_overlap, negative_slope)
        self.ge_decoder = VGAE_Decoder(output_dim_g_emb, input_dim)
        self.label_predictor = GLabelPredictor(output_dim_g_emb, output_dim)

    def forward(self, g: GData):
        x = self.msgp(g)
        x_recon = self.ge_decoder(x)
        x_label = self.label_predictor(x)
        return x, x_recon, x_label
    
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
    
    def _loss_mtl(self, x_recon: torch.Tensor, x_label: torch.Tensor, h: torch.Tensor, true_label: torch.Tensor):
        targets = {"Reconstruction": h, "Label": true_label}
        preds = {"Reconstruction": x_recon, "Label": x_label}
        metrics = MultitaskWrapper({"Reconstruction": , "Label": MeanSquaredError})
    
    def _define_metrics(self, output_dim: int, regression: bool):
        r"""Define the loss function and the metrics.
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
    
    def _loss(self, which_step: str, y: torch.Tensor, y_pred: torch.Tensor):
        #!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.output_dim > 1 and not self.is_regression:
            y = y.argmax(dim=-1)

        if self.output_dim == 1:
            loss = self.loss_fn(y_pred, y)
            if which_step == const.dkey.title_predict:
                return loss
            self.log(f"{which_step}_loss", loss, sync_dist=True)
            self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
            self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
        else:
            if self.is_regression:
                loss = self.loss_fn(y_pred, y).mean(dim=0).sum()
                if which_step == const.dkey.title_predict:
                    return loss
                self.log(f"{which_step}_loss", loss, sync_dist=True)
                # self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
                # if y.shape[0] < 2:
                #     return loss
                # self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
                # self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
            else:
                loss = self.loss_fn(y_pred, y)
                if which_step == const.dkey.title_predict:
                    return loss
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
