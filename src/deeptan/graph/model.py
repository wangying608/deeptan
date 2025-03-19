r"""
DeepTAN:
Trait-associated multi-omics network inference via multi-task NMIC-guided adaptive multi-scale graph embedding.
"""

import os
import pickle
import random
import time
from typing import Any, Dict, List, Optional

import lightning as ltn
import optuna
import polars as pl
import torch
import torch._dynamo.config
import torch.nn.functional as F
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

# from lightning.pytorch.profilers import AdvancedProfiler
from litdata import StreamingDataLoader, StreamingDataset
from torch.optim.adamw import AdamW
from torch_geometric.data import Data as GData
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
)

import deeptan.constants as const
from deeptan.constants.art import ascii_art
from deeptan.graph.modules import AMSGP, GE_Decoder, GLabelPredictor
from deeptan.utils.data import (
    DeepTANDataModule,
    DeepTANDataModuleLit,
    celltypes_class_weights,
)
from deeptan.utils.uni import collate_fn, get_map_location, random_string, time_string

print(ascii_art)
torch.set_float32_matmul_precision(const.default.matmul_precision)
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True


class FocalLoss(torch.nn.Module):
    r"""Multi-class Focal Loss
    Formula: loss = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(
        self,
        gamma: float = 0.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=None,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)  # p = exp(-CE)
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_weight = alpha[targets]
            loss = alpha_weight * loss

        return loss.mean() if self.reduction == "mean" else loss


class DeepTAN(ltn.LightningModule):
    r"""
    DeepTAN.
    """

    def __init__(
        self,
        z_dict_node_names: Dict[str, int],
        input_dim: int,
        output_g_label_dim: Optional[int],
        is_regression: bool,
        class_weights: Optional[List[float]] = None,
        use_focal_loss: bool = True,
        focal_alpha: Optional[List[float]] = None,
        node_emb_dim: int = 128,
        fusion_dims_node_emb: List[int] = [128, 64],
        output_dim_g_emb: int = 192,
        n_hop: int = const.default.n_hop,
        threshold_edge_exist: float = const.default.threshold_edge_exist,
        threshold_subgraph_overlap: float = const.default.threshold_subg_overlap,
        n_heads_node_emb: int = 2,
        n_heads_pooling: int = 2,
        n_heads_ge_decoder: int = 2,
        n_heads_label_pred: int = 2,
        dropout: float = const.default.dropout,
        lr: float = const.default.lr,
        chunk_size: int = const.default.chunk_size,
    ):
        r"""
        Initialize the DeepTAN model.

        Args:
            z_dict_node_names (Dict[str, int]): A dictionary mapping node names to their respective indices.
            input_dim (int): The dimension of the input features.
            output_g_label_dim (Optional[int]): The dimension of the output graph label. If None, it defaults to 2.
            is_regression (bool): Whether the task is a regression task.
            class_weights (Optional[List[float]]): A list of label class weights for the loss function.
            focal_alpha (Optional[List[float]]): A list of alpha values for the focal loss.
            node_emb_dim (int): The dimension of the node embeddings.
            fusion_dims_node_emb (List[int]): A list of dimensions for the fusion layers of node embeddings.
            output_dim_g_emb (int): The dimension of the output graph embeddings.
            n_hop (int): The number of hops for subgraph division.
            threshold_edge_exist (float): The threshold for edge existence.
            threshold_subgraph_overlap (float): The threshold for subgraph overlap. Similar subgraphs are merged if their overlap exceeds this threshold.
            n_heads_node_emb (int): The number of attention heads for the node embedding layers.
            n_heads_pooling (int): The number of attention heads for the pooling layers.
            n_heads_ge_decoder (int): The number of attention heads for the graph embedding decoder.
            n_heads_label_pred (int): The number of attention heads for the label predictor.
            dropout (float): The dropout rate.
            lr (float): The learning rate.
            chunk_size (int): The chunk size for processing large matrices.
        """
        super().__init__()
        self.save_hyperparameters()

        self.dict_node_names = z_dict_node_names
        self.all_node_names = list(z_dict_node_names.keys())
        self.num_all_nodes = len(self.all_node_names)
        self.register_buffer("all_node_indices", torch.arange(self.num_all_nodes, dtype=torch.long), persistent=False)

        self.chunk_size = chunk_size
        self.input_dim = input_dim
        self.is_regression = is_regression
        self.lr = lr
        self.output_g_label_dim = output_g_label_dim

        self.output_dim = output_g_label_dim if output_g_label_dim is not None else 2
        self.class_weights = class_weights

        self.use_focal_loss = use_focal_loss
        if focal_alpha is None:
            if class_weights is not None:
                self.focal_alpha = torch.tensor(class_weights)
            else:
                self.focal_alpha = torch.tensor([1.0] * self.output_dim)
        else:
            self.focal_alpha = torch.tensor(focal_alpha)
        self.focal_gamma = 0.5
        self.current_epoch_work = 0
        self.current_epoch_tmp = 0
        self.loss_smooth = 0.0  # For scaling label loss if label is not None

        # Core components
        self.amsgp = AMSGP(
            dict_node_names=self.dict_node_names,
            input_dim=input_dim,
            node_emb_dim=node_emb_dim,
            fusion_dims_node_emb=fusion_dims_node_emb,
            n_heads_node_emb=n_heads_node_emb,
            output_dim_g_emb=output_dim_g_emb,
            n_heads_pooling=n_heads_pooling,
            n_hop=n_hop,
            threshold_edge_exist=threshold_edge_exist,
            threshold_subgraph_overlap=threshold_subgraph_overlap,
            dropout=dropout,
            chunk_size=self.chunk_size,
        )

        # Multi-task decoders
        self.ge_decoder = GE_Decoder(
            z_dim=output_dim_g_emb,
            h_dim=node_emb_dim,
            output_dim=input_dim,
            hidden_dim=node_emb_dim,
            dropout=dropout,
            chunk_size=self.chunk_size,
            n_heads=n_heads_ge_decoder,
            n_res_blocks=3,
        )

        # Graph-level label predictor
        self.g_label_predictor = GLabelPredictor(
            output_dim_g_emb,
            self.output_dim,
            const.default.label_pred_hidden_dims,
            dropout,
            n_heads=n_heads_label_pred,
        )

        # Metrics and initialization
        self._init_metrics()

    def forward(self, batch: GData) -> Dict[str, Any]:
        # assert batch.x is not None, "Input x is None"
        # assert batch.edge_index is not None, "Input edge_index is None"
        # assert batch.x.dim() == 2, f"The input dim is wrong: {batch.x.shape}"
        # assert batch.edge_index.max() < batch.x.size(0), f"The edge index is wrong: {batch.edge_index.shape}"

        # Check if all node names are valid
        for nodes in batch.node_names:
            assert all(n in self.dict_node_names for n in nodes), f"Node names are not valid: {batch.node_names}"

        # Extract batch information if available, otherwise initialize with zeros
        node_batch = getattr(batch, "batch", torch.zeros(batch.x.size(0), dtype=torch.long, device=self.device))

        # Initialize scale factors
        self.scale_factors = torch.ones(2, dtype=torch.float32, device=self.device).softmax(dim=0)

        # Embedding features
        z, E_all, ids = self.amsgp(
            node_names=batch.node_names,
            x=batch.x,
            edge_attr=batch.edge_attr,
            edge_index=batch.edge_index,
            batch=node_batch,
        )

        # Reconstruct node embeddings
        recon_node_emb, recon_node_val_for_loss_all = self.ge_decoder(z, E_all)

        # Graph-level label prediction
        pred_labels = self.g_label_predictor(z)

        # Node-level reconstruction loss
        predicted_value_for_loss_nonzero = torch.zeros_like(ids, dtype=recon_node_val_for_loss_all.dtype, device=self.device)
        predicted_value_for_loss_zero = torch.zeros(recon_node_val_for_loss_all.shape[0] * recon_node_val_for_loss_all.shape[1] - len(ids.flatten()), dtype=recon_node_val_for_loss_all.dtype, device=self.device)

        # For each batch, fill the mask for available nodes
        sta_nonezero = 0
        sta_zero = 0
        for i, nodes in enumerate(batch.node_names):
            n_nonezero = len(nodes)
            n_zero = recon_node_val_for_loss_all.shape[1] - n_nonezero
            node_indices = ids[sta_nonezero : (sta_nonezero + n_nonezero)]
            mask_ = torch.zeros(recon_node_val_for_loss_all.shape[1], dtype=torch.bool, device=self.device)
            mask_[node_indices] = True

            predicted_value_for_loss_nonzero[sta_nonezero : (sta_nonezero + n_nonezero)] = recon_node_val_for_loss_all[i, node_indices].flatten()
            predicted_value_for_loss_zero[sta_zero : (sta_zero + n_zero)] = recon_node_val_for_loss_all[i, ~mask_].squeeze()
            sta_nonezero += n_nonezero
            sta_zero += n_zero

        return {
            "embedding": z,
            "node_recon": recon_node_emb,
            "node_recon_for_loss_all": recon_node_val_for_loss_all,
            "label_pred": pred_labels,
            "node_recon_for_loss": predicted_value_for_loss_nonzero,
            "node_recon_for_loss_zeros": predicted_value_for_loss_zero,
        }

    def training_step(self, batch: GData, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: GData, batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch: GData, batch_idx: int):
        return self._shared_step(batch, "test")

    def predict_step(self, batch: GData, batch_idx: int):
        return self(batch)

    def _shared_step(self, batch: GData, stage: str) -> torch.Tensor:
        outputs = self(batch)
        losses = self._compute_losses(outputs, batch, stage)

        # More loss computation
        if batch.y is not None:
            preds = outputs["label_pred"]
            targets = torch.as_tensor(batch.y, device=preds.device)
            if not self.is_regression:
                if targets.ndim > 1 and targets.shape[1] > 1:
                    targets = torch.argmax(targets, dim=1)
            self.metrics_task_label[f"{stage}_metrics"].update(preds, targets)

        # Logging metrics and losses
        self._log_metrics(losses, stage)

        # del outputs
        torch.cuda.empty_cache()

        return losses["loss"]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": const.dkey.title_val_loss,
            },
        }

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < 2:
            # Initially focused on overall distribution
            self.focal_gamma = 0.5
        else:
            # Focus on difficult samples in the later stage
            self.focal_gamma = 2.0

    def _init_metrics(self):
        if hasattr(self, "metrics"):
            self.metrics.clear()

        metrics_common = MetricCollection(
            {
                "MSE": MeanSquaredError(num_outputs=self.input_dim),
                "MAE": MeanAbsoluteError(num_outputs=self.input_dim),
                "PCC": PearsonCorrCoef(num_outputs=self.input_dim),
                # "KLD": KLDivergence(),
                # "JSD": self._js_divergence,
            }
        )
        if self.is_regression:
            metrics_task_label = MetricCollection(
                {
                    "MSE": MeanSquaredError(num_outputs=self.output_dim),
                    "MAE": MeanAbsoluteError(num_outputs=self.output_dim),
                    "PCC": PearsonCorrCoef(num_outputs=self.output_dim),
                    "RMSE": MeanSquaredError(num_outputs=self.output_dim, squared=False),
                }
            )
        else:
            metrics_task_label = MetricCollection(
                {
                    "F1_weighted": MulticlassF1Score(num_classes=self.output_dim, average="weighted"),
                    "F1_macro": MulticlassF1Score(num_classes=self.output_dim, average="macro"),
                    "F1_micro": MulticlassF1Score(num_classes=self.output_dim, average="micro"),
                    "Accuracy": MulticlassAccuracy(num_classes=self.output_dim),
                    "Precision": MulticlassPrecision(num_classes=self.output_dim, average="weighted"),
                    "Recall": MulticlassRecall(num_classes=self.output_dim, average="weighted"),
                    "AUROC": MulticlassAUROC(num_classes=self.output_dim, average="macro"),
                }
            )

        # Create metrics for all stages
        self.metrics_common = torch.nn.ModuleDict({f"{k}_metrics": metrics_common.clone(prefix=k + "/recon_") for k in ["train", "val", "test"]})
        self.metrics_task_label = torch.nn.ModuleDict({f"{k}_metrics": metrics_task_label.clone(prefix=k + "/label_") for k in ["train", "val", "test"]})

    def _compute_losses(self, outputs: Dict, batch: GData, stage: str) -> Dict:
        # assert batch.x is not None
        losses = {}

        node_true_val_for_loss = batch.x.squeeze(1)

        self.metrics_common[f"{stage}_metrics"].update(outputs["node_recon_for_loss"], node_true_val_for_loss)

        # Node reconstruction loss
        recon_loss = F.mse_loss(outputs["node_recon_for_loss"], node_true_val_for_loss)

        # kl_loss = F.kl_div(
        #     F.log_softmax(outputs["node_recon_for_loss"], dim=1),
        #     F.softmax(node_true_val_for_loss, dim=1),
        #     log_target=True,
        #     reduction="mean",
        # )

        recon_loss_zeros = F.mse_loss(
            outputs["node_recon_for_loss_zeros"],
            torch.zeros_like(outputs["node_recon_for_loss_zeros"]),
        )

        # losses["recon_KLD"] = kl_loss
        losses["recon_MSE"] = recon_loss
        losses["recon_zeros"] = recon_loss_zeros

        # Total reconstruction loss
        # losses["recon"] = recon_loss + kl_loss + recon_loss_zeros
        losses["recon"] = recon_loss + recon_loss_zeros

        # Graph-level label prediction loss
        if batch.y is None:
            # Placeholder for no label loss
            losses["label"] = torch.tensor(0.0, device=self.device)
        else:
            if isinstance(batch.y, torch.Tensor):
                _y = batch.y
            else:
                _y = torch.tensor(batch.y, device=self.device)

            if self.is_regression:
                pred_loss = F.mse_loss(outputs["label_pred"], _y)
            else:
                if _y.ndim > 1 and _y.shape[1] > 1:
                    _y = torch.argmax(_y, dim=1)

                if self.use_focal_loss:
                    focal_loss = FocalLoss(gamma=self.focal_gamma, alpha=self.focal_alpha, reduction="mean")
                    pred_loss = focal_loss(outputs["label_pred"], _y)
                else:
                    if self.class_weights is None:
                        pred_loss = F.cross_entropy(outputs["label_pred"], _y)
                    else:
                        pred_loss = F.cross_entropy(
                            outputs["label_pred"],
                            _y,
                            weight=torch.tensor(
                                self.class_weights,
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        )

            losses["label"] = pred_loss

        # Dynamic weight adjustment
        total_loss, unweighted_loss = self._balance_losses(losses, stage)
        losses["loss_unweighted"] = unweighted_loss
        losses["loss"] = total_loss
        losses["loss_smooth"] = self.loss_smooth
        return losses

    def _balance_losses(self, losses: Dict, stage: str):
        # If in the initial epochs, focus only on reconstruction loss
        # if stage == "train" and self.current_epoch < 1:
        #     self.current_epoch_tmp = self.current_epoch
        #     total_loss = losses["recon"]

        if self.current_epoch_tmp < self.current_epoch:
            self.current_epoch_tmp = self.current_epoch
            self.current_epoch_work = self.current_epoch_work + 1
            if self.current_epoch_work > 2:
                self.current_epoch_work = 0

            # if self.current_epoch_work == 0:
            #     # Focus on reconstruction loss
            #     self.g_label_predictor.requires_grad_(False)
            #     self.ge_decoder.requires_grad_(True)
            #     self.amsgp.requires_grad_(True)
            # elif self.current_epoch_work == 1:
            #     # Focus on label prediction loss
            #     self.g_label_predictor.requires_grad_(True)
            #     self.ge_decoder.requires_grad_(False)
            #     self.amsgp.requires_grad_(False)
            # else:
            #     # Mix both losses
            #     self.g_label_predictor.requires_grad_(True)
            #     self.ge_decoder.requires_grad_(True)
            #     self.amsgp.requires_grad_(True)

            # self.zero_grad(set_to_none=False)

        if stage == "train":
            if self.current_epoch_work == 0:
                # Focus on reconstruction loss
                total_loss = losses["recon"]

            elif self.current_epoch_work == 1:
                # Focus on label prediction loss
                total_loss = losses["label"] / (losses["label"] / self.loss_smooth).detach()

            else:
                # Dynamic loss scaling
                # EMA?
                # loss_ratio = torch.stack([losses["label"].detach(), losses["recon"].detach()])
                # rel_ratio = loss_ratio / (loss_ratio.mean() + 1e-8)
                # _scale_factors = 1.0 / rel_ratio
                # # _scale_factors = F.softmax(_scale_factors, dim=0)

                # alpha = 0.1
                # self.scale_factors = alpha * _scale_factors + (1.0 - alpha) * self.scale_factors

                # # Calculate total loss with scaling
                # total_loss = losses["label"] * self.scale_factors[0] + losses["recon"] * self.scale_factors[1]
                total_loss = 0.5 * losses["recon"] + 0.5 * losses["label"] / (losses["label"] / self.loss_smooth).detach()

            self.loss_smooth = 0.2 * total_loss.detach() + 0.8 * self.loss_smooth

        else:
            total_loss = 0.5 * losses["recon"] + 0.5 * losses["label"] / (losses["label"] / self.loss_smooth).detach()

        unweighted_loss = 0.5 * losses["recon"] + 0.5 * losses["label"]

        return total_loss, unweighted_loss

    def on_before_optimizer_step(self, optimizer):
        self.log("scale_factors/label", self.scale_factors[0])
        self.log("scale_factors/recon", self.scale_factors[1])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("loss_params/focal_gamma", self.focal_gamma)
        self.log("loss_params/focal_alpha_mean", self.focal_alpha.mean())

    def _log_metrics(self, losses: Dict, stage: str):
        for k, v in losses.items():
            self.log(
                f"{stage}/{k}",
                v,
                prog_bar=(k == "loss"),
                sync_dist=True,
                batch_size=self._get_batch_size(stage),
            )

        # Log evaluation metrics
        if not self.trainer.sanity_checking:
            metrics_common = self.metrics_common[f"{stage}_metrics"].compute()
            for name, val in metrics_common.items():
                self.log(name, val, sync_dist=True, batch_size=self._get_batch_size(stage))
            self.metrics_common[f"{stage}_metrics"].reset()

            if self.output_g_label_dim is not None:
                metrics_task_label = self.metrics_task_label[f"{stage}_metrics"].compute()
                for name, val in metrics_task_label.items():
                    self.log(name, val, sync_dist=True, batch_size=self._get_batch_size(stage))
                self.metrics_task_label[f"{stage}_metrics"].reset()

    def _get_batch_size(self, stage: str) -> int:
        if stage == "train":
            return self.trainer.train_dataloader.batch_size if self.trainer.train_dataloader is not None else 1
        elif stage == "val":
            return self.trainer.val_dataloaders.batch_size if self.trainer.val_dataloaders is not None else 1
        elif stage == "test":
            return self.trainer.test_dataloaders.batch_size if self.trainer.test_dataloaders is not None else 1
        return 1

    def save_components(self, save_dir: str):
        """
        Save core components separately
        """
        os.makedirs(save_dir, exist_ok=True)
        components = {
            "amsgp": self.amsgp,
            "ge_decoder": self.ge_decoder,
            "g_label_predictor": self.g_label_predictor,
        }
        for name, module in components.items():
            torch.save(
                {"state_dict": module.state_dict()},
                os.path.join(save_dir, f"{name}.pt"),
            )

    @classmethod
    def load_component(cls, ckpt_path: str, target_class: Any):
        """
        Load a specific component
        """
        ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        instance = target_class.__new__(target_class)
        instance.load_state_dict(ckpt["state_dict"])
        return instance


def train_model(
    model: Any,
    datamodule: LightningDataModule,
    es_patience: int,
    max_epochs: int,
    min_epochs: int,
    log_dir: str,
    accumulate_grad_batches: int = const.default.accumulate_grad_batches,
    accelerator: str = const.default.accelerator,
    fast_dev_run: bool = False,
):
    r"""Fit the model.

    Args:
        model (Any): The model to train.
        datamodule (LightningDataModule): The data module.
        es_patience (int): The patience for early stopping.
        max_epochs (int): The maximum number of epochs.
        min_epochs (int): The minimum number of epochs.
        log_dir (str): The directory to log the training results.
        accumulate_grad_batches (int): The number of batches to accumulate gradients over.
        accelerator (str): The accelerator to use.
        fast_dev_run (bool): Whether to run a fast development run.
    """

    # torch.autograd.set_detect_anomaly(True)

    callback_es = EarlyStopping(
        # monitor=const.dkey.title_val_loss,
        monitor="val/loss_unweighted",
        patience=es_patience,
        mode="min",
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename=const.default.ckpt_fname_format,
        # monitor=const.dkey.title_val_loss,
        monitor="val/loss_unweighted",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger_tr = TensorBoardLogger(save_dir=log_dir, name="")

    # profiler = AdvancedProfiler(dirpath=log_dir, filename="perf_logs")

    trainer = Trainer(
        fast_dev_run=fast_dev_run,
        # profiler=profiler,
        # strategy="ddp_spawn",
        enable_progress_bar=True,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger_tr,
        log_every_n_steps=1,
        precision="16-mixed",
        accelerator=accelerator,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[callback_es, callback_ckpt, lr_monitor],
        num_sanity_val_steps=0,
        default_root_dir=log_dir,
        gradient_clip_val=3.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model=model, datamodule=datamodule)

    if callback_ckpt.best_model_score is not None:
        best_score = callback_ckpt.best_model_score.item()
    else:
        best_score = None

    trainer.test(ckpt_path=callback_ckpt.best_model_path, dataloaders=datamodule)

    print(f"\nBest validation score: {best_score}")
    print(f"Best model path: {callback_ckpt.best_model_path}\n")

    return best_score


class DeepTANTune:
    r"""
    DeepTAN hyperparameter tuning class with Optuna integration.
    """

    def __init__(self, args: Dict[str, Any], existing_model_path: Optional[str] = None):
        """Initialize tuning environment with parameters."""
        # Store configuration parameters
        self.args = args
        self.existing_model_path = existing_model_path

        self.is_regression = self.args["is_regression"]
        self.log_dir = self.args["log_dir"]
        if self.log_dir.endswith("/"):
            self.log_dir = self.log_dir[:-1]

        self.log_name = f"DeepTAN_{time_string()}_{random_string(5)}"
        os.makedirs(os.path.join(self.log_dir, self.log_name), exist_ok=True)
        self.path_optuna_db = "sqlite:///" + self.log_dir + f"/{self.log_name}/optuna.db"

        # Initialize data module
        self._init_data_module()

        # Initialize class weights
        if not self.is_regression:
            self.class_weight = self._init_class_weights()
        else:
            self.class_weight = None

    def _init_data_module(self):
        """Initialize data module based on input parameters."""
        if self.args.get("litdata"):
            with open(os.path.join(self.args["litdata"], const.fname.litdata_others2save_pkl), "rb") as f:
                others2save = pickle.load(f)
            self.dict_node_names = others2save["dict_node_names"]
            self.output_g_label_dim = others2save["output_g_label_dim"]

            path_label_onehot = os.path.join(self.args["litdata"], const.fname.label_class_onehot)
            self.path_label_onehot = path_label_onehot if os.path.exists(path_label_onehot) else None

            self.datamodule = DeepTANDataModuleLit(
                self.args["litdata"],
                batch_size=self.args["bs"],
                n_workers=self.args["nworker"],
            )
            self.datamodule.setup()

        elif all([self.args.get(k) for k in ["trn_npz", "val_parquet", "tst_parquet"]]):
            labels = self.args["labels"] if self.args.get("labels") else None
            files_fit = {
                "trn": self.args["trn_npz"],
                "val": self.args["val_parquet"],
                "tst": self.args["tst_parquet"],
            }
            self.datamodule = DeepTANDataModule(files_fit, labels, batch_size=self.args["bs"])
            self.datamodule.setup()
            self.dict_node_names = self.datamodule.dict_node_names
            self.output_g_label_dim = self.datamodule.label_dim

            path_label_onehot = os.path.join(os.path.dirname(self.args["val_parquet"]), const.fname.label_class_onehot)
            self.path_label_onehot = path_label_onehot if os.path.exists(path_label_onehot) else None

        else:
            raise ValueError("Invalid data configuration")

    def _init_class_weights(self):
        """Initialize class weights if provided."""
        if self.path_label_onehot is not None:
            print("\nPre-defined label onehot file found. Computing class weights...\n")
            return celltypes_class_weights(pl.read_parquet(self.path_label_onehot))
        print(f"\nNo pre-defined label onehot file ( {self.path_label_onehot} ) found. Skipping class weights computation...\n")
        return None

    def create_model(self, trial_params: Dict[str, Any]) -> DeepTAN:
        """Create model instance with trial parameters."""
        fusion_dims_node_emb = trial_params.get("fusion_dims_node_emb", self.args["fusion_dims_node_emb"])
        if isinstance(fusion_dims_node_emb, str):
            fusion_dims_node_emb = eval(fusion_dims_node_emb)

        if self.existing_model_path is not None:
            _amsgp, _ge_decoder = self._load_ckpt(self.existing_model_path, self.dict_node_names)
            output_dim_g_emb = _amsgp.output_dim_g_emb
        else:
            output_dim_g_emb = trial_params.get("output_dim_g_emb", self.args["output_dim_g_emb"])

        _model = DeepTAN(
            z_dict_node_names=self.dict_node_names,
            input_dim=self.args["input_node_emb_dim"],
            output_g_label_dim=self.output_g_label_dim,
            is_regression=self.is_regression,
            class_weights=self.class_weight,
            node_emb_dim=trial_params.get("node_emb_dim", self.args["node_emb_dim"]),
            fusion_dims_node_emb=fusion_dims_node_emb,
            output_dim_g_emb=output_dim_g_emb,
            n_hop=trial_params.get("n_hop", self.args["n_hop"]),
            threshold_edge_exist=trial_params.get("threshold_edge_exist", self.args["threshold_edge_exist"]),
            threshold_subgraph_overlap=trial_params.get("threshold_subgraph_overlap", self.args["threshold_subgraph_overlap"]),
            n_heads_node_emb=trial_params.get("n_heads_node_emb", self.args["n_heads_node_emb"]),
            n_heads_pooling=trial_params.get("n_heads_pooling", self.args["n_heads_pooling"]),
            n_heads_ge_decoder=trial_params.get("n_heads_ge_decoder", self.args["n_heads_ge_decoder"]),
            n_heads_label_pred=trial_params.get("n_heads_label_pred", self.args["n_heads_label_pred"]),
            dropout=trial_params.get("dropout", self.args["dropout"]),
            lr=trial_params.get("lr", self.args["lr"]),
            chunk_size=self.args["chunk_size"],
        )

        if self.existing_model_path is not None:
            _model.amsgp = _amsgp
            _model.ge_decoder = _ge_decoder

        return _model

    def _init_model(self):
        """This function is used for create model directly without getting ``trial_params``."""

        if self.existing_model_path is not None:
            _amsgp, _ge_decoder = self._load_ckpt(self.existing_model_path, self.dict_node_names)
            output_dim_g_emb = _amsgp.output_dim_g_emb
        else:
            output_dim_g_emb = self.args["output_dim_g_emb"]

        _model = DeepTAN(
            z_dict_node_names=self.dict_node_names,
            input_dim=self.args["input_node_emb_dim"],
            output_g_label_dim=self.output_g_label_dim,
            is_regression=self.args["is_regression"],
            class_weights=self.class_weight,
            node_emb_dim=self.args["node_emb_dim"],
            fusion_dims_node_emb=self.args["fusion_dims_node_emb"],
            output_dim_g_emb=output_dim_g_emb,
            n_hop=self.args["n_hop"],
            threshold_edge_exist=self.args["threshold_edge_exist"],
            threshold_subgraph_overlap=self.args["threshold_subgraph_overlap"],
            n_heads_node_emb=self.args["n_heads_node_emb"],
            n_heads_pooling=self.args["n_heads_pooling"],
            n_heads_ge_decoder=self.args["n_heads_ge_decoder"],
            n_heads_label_pred=self.args["n_heads_label_pred"],
            dropout=self.args["dropout"],
            lr=self.args["lr"],
            chunk_size=self.args["chunk_size"],
        )

        if self.existing_model_path is not None:
            _model.amsgp = _amsgp
            _model.ge_decoder = _ge_decoder

        return _model

    def _load_ckpt(self, existing_model_path: str, dict_node_names_new: Dict[str, int]):
        r"""
        Load a checkpoint from the given path and extract its graph embedding and decoding module.
        """
        _model_pre = DeepTAN.load_from_checkpoint(existing_model_path, map_location=get_map_location())

        # Extract AMSGP modules
        _model_amsgp = _model_pre.amsgp
        _model_ge_decoder = _model_pre.ge_decoder
        # print(_model_amsgp)

        dict_node_names_former = _model_amsgp.node_embedding_layers.dict_node_names
        # print("\n", dict_node_names_former)
        if set(dict_node_names_new.keys()) != set(dict_node_names_former.keys()):
            print("\nUpdating dict_node_names in NodeEmbedding")
            new_nodes_to_append = set(dict_node_names_new.keys()) - set(dict_node_names_former.keys())
            n_node_former = len(dict_node_names_former)
            n_node_add = len(new_nodes_to_append)
            new_node_num = n_node_former + n_node_add
            dict_to_add = {node: n_node_former + i for i, node in enumerate(new_nodes_to_append)}
            print(f"The feature embedding module is extended from {n_node_former} to {new_node_num} with {n_node_add} new features.\n")

            # Update dict_node_names in NodeEmbedding
            dict_node_names_former.update(dict_to_add)
            _model_amsgp.node_embedding_layers.dict_node_names = dict_node_names_former

            # Update self.dict_node_names
            self.dict_node_names = dict_node_names_former

            # Update node embedding weights by concatenating new weights
            emb_dim = _model_amsgp.node_embedding_layers.embed.weight.size(1)
            new_embed = torch.nn.Embedding(new_node_num, emb_dim, scale_grad_by_freq=True, sparse=True)
            new_embed.weight.data[:n_node_former] = _model_amsgp.node_embedding_layers.embed.weight.data
            torch.nn.init.xavier_uniform_(new_embed.weight.data[n_node_former:])
            _model_amsgp.node_embedding_layers.embed = new_embed

        else:
            print("\ndict_node_names in NodeEmbedding is the same")

        return _model_amsgp, _model_ge_decoder

    def _train_on_args(self):
        """This function is used for training the model with the given arguments."""
        _model = self._init_model()

        train_model(
            model=_model,
            datamodule=self.datamodule,
            es_patience=self.args["es"],
            max_epochs=self.args["max_ep"],
            min_epochs=self.args["min_ep"],
            log_dir=os.path.join(self.log_dir, self.log_name),
            accumulate_grad_batches=self.args["acc_grad_batch"],
            accelerator=self.args["accelerator"],
            # fast_dev_run=True,
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""

        time_delay = const.default.time_delay * random.uniform(0.3, 1.1)
        print(f"\nWaiting for {time_delay} seconds...\n")
        time.sleep(time_delay)
        print(f"Starting trial number: {trial.number}\n")

        try:
            fusion_dims_node_emb_lists_to_try = [[128, 64], [64, 32]]
            fusion_dims_node_emb_list_strings = [str(lst) for lst in fusion_dims_node_emb_lists_to_try]

            # Suggest hyperparameters
            params = {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.6, step=0.2),
                "node_emb_dim": trial.suggest_categorical("node_emb_dim", [128, 192, 256]),
                "n_heads_node_emb": trial.suggest_categorical("n_heads_node_emb", [2, 4]),
                "n_heads_pooling": trial.suggest_categorical("n_heads_pooling", [2, 4]),
                "n_heads_ge_decoder": trial.suggest_categorical("n_heads_ge_decoder", [2, 4]),
                "n_heads_label_pred": trial.suggest_categorical("n_heads_label_pred", [2, 4]),
                "fusion_dims_node_emb": trial.suggest_categorical("fusion_dims_node_emb", fusion_dims_node_emb_list_strings),
                "output_dim_g_emb": trial.suggest_categorical("output_dim_g_emb", [128, 192, 256, 512]),
                "n_hop": trial.suggest_int("n_hop", 1, 2),
            }

            # Create model with suggested parameters
            _model = self.create_model(params)

            # Execute training run
            val_loss = train_model(
                model=_model,
                datamodule=self.datamodule,
                es_patience=self.args["es"],
                max_epochs=self.args["max_ep"],
                min_epochs=self.args["min_ep"],
                log_dir=os.path.join(self.log_dir, self.log_name, f"trial_{trial.number}"),
                accumulate_grad_batches=self.args["acc_grad_batch"],
                accelerator=self.args["accelerator"],
            )

            if val_loss is None:
                print(f"\n\nThe validation loss for trial {trial.number} is None. Skipping trial.\n")
                raise optuna.TrialPruned()

            return val_loss

        except torch.cuda.OutOfMemoryError:
            print("\n\nOut of memory error, skipping trial\n")
            raise optuna.TrialPruned()
        except Exception as e:
            print(f"\n\nAn error occurred in trial {trial.number}: {e}\n")
            raise optuna.TrialPruned()

    def optimize(self, n_trials: int = 100, n_jobs: int = 1):
        """Run hyperparameter optimization with Optuna"""
        study = optuna.create_study(
            direction="minimize",
            study_name=self.log_name,
            storage=self.path_optuna_db,
            load_if_exists=True,
        )

        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")


def predict(
    model_ckpt_path: str,
    litdata_dir: str,
    output_pickle_path: str,
    map_location: Optional[str] = None,
    batch_size: int = 1,
):
    # Load a DeepTAN model
    model = DeepTAN.load_from_checkpoint(model_ckpt_path, map_location=get_map_location(map_location))
    # Freeze the model
    model.eval()
    model.freeze()

    # Load the LitData dataset
    dataloader = StreamingDataLoader(StreamingDataset(litdata_dir), batch_size=batch_size, collate_fn=collate_fn)

    # Predict
    trainer = Trainer(logger=False)
    results = trainer.predict(model=model, dataloaders=dataloader)

    assert results is not None
    # Save the results to a pickle file
    with open(output_pickle_path, "wb") as f:
        pickle.dump(results, f)


def process_results(pickle_path: str, output_pkl: str):
    # Load the results
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    g_embedding = []
    node_recon = []
    node_recon_for_loss = []
    node_recon_all = []
    labels = []

    for i_batch in range(len(results)):
        g_embedding.append(results[i_batch]["embedding"])
        node_recon.append(results[i_batch]["node_recon"])
        node_recon_for_loss.append(results[i_batch]["node_recon_for_loss"])
        node_recon_all.append(results[i_batch]["node_recon_for_loss_all"])
        labels.append(results[i_batch]["label_pred"])

    g_embedding = torch.cat(g_embedding, dim=0)
    node_recon = torch.cat(node_recon, dim=0)
    node_recon_all = torch.cat(node_recon_all, dim=0)
    labels = torch.cat(labels, dim=0)

    # Convert to numpy arrays for further processing
    g_embedding_np = g_embedding.detach().cpu().numpy()
    node_recon_np = node_recon.detach().cpu().numpy()
    node_recon_all_np = node_recon_all.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Save the results as a dictionary in a pickle file
    results_dict = {
        "g_embedding": g_embedding_np,
        "node_recon": node_recon_np,
        "node_recon_all": node_recon_all_np,
        "labels": labels_np,
    }

    print(results_dict.keys())
    # For each key in the results dictionary, print data shape
    for key in results_dict.keys():
        print(f"Key: {key}, Shape: {results_dict[key].shape}")

    if not output_pkl.endswith(".pkl"):
        output_pkl += ".pkl"
    print(f"Saving results to {output_pkl}")
    with open(output_pkl, "wb") as f:
        pickle.dump(results_dict, f)
