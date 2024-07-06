r"""

"""
import torch
from torch import set_float32_matmul_precision, nn, Tensor, optim
from lightning import LightningModule, seed_everything
# import numpy as np
# import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torch_geometric.nn import GATv2Conv, GatedGraphConv


seed_everything(2024)
set_float32_matmul_precision('medium')


class PooledGAT(LightningModule):
    """
    """
    def __init__(
            self,
            # output_dim: int,
            node_dim: int,
            heads: int,
            dropout: float,
        ):
        super().__init__()
        # self.conv1 = GCNConv(dataset.num_node_features, 16)
        # self.conv2 = GCNConv(16, dataset.num_classes)

        self.gat_features = GATv2Conv(
            in_channels=1,
            out_channels=node_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
        )
        # self.gat_output = GATv2Conv(
        #     in_channels=node_dim,
        #     out_channels=1,
        #     heads=heads,
        #     dropout=dropout,
        # )

        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # x, edge_index, attention_weights = self.gat_features(x, edge_index, edge_attr, True)
        x = self.gat_features(x, edge_index, edge_attr)
        # x = self.gat_output(x, edge_index)

        # out = nn.functional.log_softmax(x, dim=1)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'monitor': 'val_loss',
                    }
                }
    
    def training_step(self, batch, batch_idx):
        print(batch.keys())
        # loss, acc = self.forward(batch)
        # self.log("train_loss", loss)
        # self.log("train_acc", acc)
        # return loss


class MyTrainer:
    r"""

    """
    def __init__(
            self,

        ):
        """
        """