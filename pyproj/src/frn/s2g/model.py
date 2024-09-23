r"""
This code aims to reduce the dimensionality of SNPs, assumming that SNPs are located in genome regions.
The SNP-genome block relation is pre-defined.
The input is the one-hot SNPs, and the first layer of the network is a sparse linear layer that maps the SNPs to a low-dimensional space with features representing genome regions.
The following layers are dense layers, that could be trained to predict phenotypes based on the low-dimensional features.
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import lightning as ltn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, R2Score, PearsonCorrCoef
from frn.utils.uni import idx_convert, get_map_location


torch.set_float32_matmul_precision('high')


class SNPReductionNetModel(nn.Module):
    def __init__(
            self,
            output_dim: int,
            blocks_gt: List[List[int]],
            len_one_hot_vec: int,
            dense_layers_hidden_dims: List[int],
        ):
        super().__init__()
        n_blocks = len(blocks_gt)
        self.n_blocks = n_blocks
        
        self.sparse_layers = nn.ModuleList()
        # Define the sparse linear layers that maps SNPs to genome blocks
        for i_gb in range(n_blocks):
            self.sparse_layers.append(nn.Linear(len(blocks_gt[i_gb]) * len_one_hot_vec, 1, bias=False))
        
        indices_gt = []
        for i_gb in range(n_blocks):
            indices_gt.append(idx_convert(blocks_gt[i_gb], len_one_hot_vec))
        self.indices_gt = indices_gt

        # Define the dense layers for predicting the phenotype
        self.dense_layers = nn.ModuleList()
        
        # Apply LayerNorm to the input features.
        self.dense_layers.append(nn.LayerNorm(n_blocks))
        
        # First dense layer takes the genome blocks features as input.
        self.dense_layers.append(nn.Linear(n_blocks, dense_layers_hidden_dims[0]))
        for i_dim in range(len(dense_layers_hidden_dims) - 1):
            self.dense_layers.append(nn.Linear(dense_layers_hidden_dims[i_dim], dense_layers_hidden_dims[i_dim + 1]))
            self.dense_layers.append(nn.Sigmoid())
            # self.dense_layers.append(nn.Dropout(p=0.1))
        self.dense_layers.append(nn.Linear(dense_layers_hidden_dims[-1], output_dim))
        # if output_dim > 1:
        #     self.dense_layers.append(nn.Softmax(dim=1))
    
    def forward(self, x):
        # Map SNPs to genome features
        g_features: list[torch.Tensor] = []
        for i_gb in range(self.n_blocks):
            g_features.append(self.sparse_layers[i_gb](x[:, self.indices_gt[i_gb]]))
        
        gblocks = torch.cat(g_features, dim=1)
        
        # Predict phenotype based on the low-dimensional features
        for layer in self.dense_layers:
            gblocks = layer(gblocks)
        
        # Return predicted phenotype(s)
        return gblocks#.type(torch.float32)


class SNPReductionNet(ltn.LightningModule):
    """
    A PyTorch Lightning module for SNP reduction and phenotype prediction.
    """
    def __init__(
            self,
            output_dim: int,
            blocks_gt: List[List[int]],
            len_one_hot_vec: int,
            dense_layers_hidden_dims: List[int],
            learning_rate: float,
            regression: bool,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.regression = regression
        
        self._define_metrics(output_dim, self.regression)

        self.model = SNPReductionNetModel(
            output_dim=output_dim,
            blocks_gt=blocks_gt,
            len_one_hot_vec=len_one_hot_vec,
            dense_layers_hidden_dims=dense_layers_hidden_dims,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['snp']
        y = batch['label']

        y_pred = self.forward(x)
        loss = self._my_loss(y_pred, y, "train", self.regression)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['snp']
        y = batch['label']
        
        y_pred = self.forward(x)
        loss = self._my_loss(y_pred, y, "val", self.regression)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['snp']
        y = batch['label']
        
        y_pred = self.forward(x)
        loss = self._my_loss(y_pred, y, "test", self.regression)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch['snp']
        y_pred = self.forward(x)
        return y_pred
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
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
                # self.r2 = R2Score()
                # self.pcc = PearsonCorrCoef()
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
                # self.log(f"{which_step}_pcc", self.pcc(y_hat, y), sync_dist=True)
                # self.log(f"{which_step}_r2", self.r2(y_hat, y), sync_dist=True)
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


class SNP2GB(ltn.LightningModule):
    """
    Transform SNPs to genome blocks using a pre-trained model.
    """
    def __init__(
            self,
            path_pretrained_model: str,
            blocks_gt: List[List[int]],
            len_one_hot_vec: int,
            map_location: Optional[str] = None,
        ):
        super().__init__()
        self.n_gb = len(blocks_gt)

        indices_gt = []
        for i_gb in range(self.n_gb):
            indices_gt.append(idx_convert(blocks_gt[i_gb], len_one_hot_vec))
        self.indices_gt = indices_gt

        # Load the pre-trained model
        pretrained_model = SNPReductionNet.load_from_checkpoint(
            checkpoint_path = path_pretrained_model,
            map_location = get_map_location(map_location),
        )
        pretrained_model.eval()
        pretrained_model.freeze()
        
        # Extract the sparse layer
        # self.sparse_layer = list(pretrained_model.children())[0]
        self.sparse_layer = pretrained_model.model.sparse_layers
        
        # Freeze the sparse layer
        self.sparse_layer.requires_grad_(False)

    def forward(self, x):
        # Map SNPs to genome blocks
        g_features: list[torch.Tensor] = []
        for i_gb in range(self.n_gb):
            g_features.append(self.sparse_layer[i_gb](x[:, self.indices_gt[i_gb]].float()))
        gblocks = torch.cat(g_features, dim=1)
        return gblocks
    
    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch['snp'])

