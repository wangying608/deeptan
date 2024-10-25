r"""
This file contains the implementation of the Pooled-GAT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
# from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from multiprocessing import cpu_count
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from .core_pgat import GATLayer

torch.set_float32_matmul_precision('high')


class ResPGAT(L.LightningModule):
    def __init__(
            self,
            in_features: int,
            out_class: int,
            n_heads: int = 4,
            dropout: float = 0.4,
            leaky_relu_slope: float = 0.1,
        ):
        super().__init__()
        self.in_features = in_features
        self.out_class = out_class
        self.n_heads = n_heads
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope

        node_features = 32
        self.pgat1 = GATLayer(in_features, node_features, n_heads, dropout, leaky_relu_slope)

        # Pooling layer
        # self.pool = nn.AvgPool1d(node_features)

    def forward(self, x, adj):
        h = self.pgat1(x, adj)
        return F.log_softmax(h, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, adj, y = batch
        sample_outputs = []
        for i in range(x.shape[0]):
            sample_outputs.append(self(x[i], adj[i]))
        output = torch.stack(sample_outputs)
        # print(output.shape)# torch.Size([8, 53, 32])
        # print(y.shape)# torch.Size([8, 1])
        # raise Exception("stop")
        
        loss = F.l1_loss(output, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer

