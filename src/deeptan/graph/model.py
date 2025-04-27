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
    JensenShannonDivergence,
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
from deeptan.utils.uni import get_map_location, random_string, time_string

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
        class_weights: Optional[List[float]],
        node_emb_dim: int,
        fusion_dims_node_emb: List[int],
        output_dim_g_emb: int,
        n_hop: int,
        threshold_edge_exist: float,
        threshold_subgraph_overlap: float,
        n_heads_node_emb: int,
        n_heads_pooling: int,
        n_heads_ge_decoder: int,
        n_heads_label_pred: int,
        dropout: float,
        lr: float,
        chunk_size: int,
        focus_task: Optional[str] = None,
        guide_gat: bool = True,
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
            focus_task (Optional[str]): Focus of the task (choose from `None`, `'recon'`, `'label'`).
            guide_gat: (Optional[bool]): Whether to apply edge weights of guidance graphs to graph attentions.
        """
        super().__init__()
        self.save_hyperparameters()

        self.dict_node_names = z_dict_node_names
        self.all_node_names = list(z_dict_node_names.keys())
        self.num_all_nodes = len(self.all_node_names)
        self.register_buffer("all_node_indices", torch.arange(self.num_all_nodes, dtype=torch.long), persistent=False)
        self.focus_task = focus_task
        self.guide_gat = guide_gat

        self.chunk_size = chunk_size
        self.input_dim = input_dim
        self.is_regression = is_regression
        self.lr = lr
        self.output_g_label_dim = output_g_label_dim

        self.output_dim = output_g_label_dim if output_g_label_dim is not None else 2
        self.class_weights = class_weights

        self.use_focal_loss = True
        focal_alpha = None
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
        # Extract batch information if available, otherwise initialize with zeros
        node_batch = getattr(batch, "batch", torch.zeros(batch.x.size(0), dtype=torch.long, device=self.device))

        # Initialize scale factors
        self.scale_factors = torch.ones(2, dtype=torch.float32, device=self.device).softmax(dim=0)

        # Embedding features
        z, E_all, ids = self.amsgp(
            node_names=batch.node_names,
            x=batch.x,
            edge_attr=batch.edge_attr if self.guide_gat else None,
            edge_index=batch.edge_index,
            batch=node_batch,
        )

        # Reconstruct node embeddings
        recon_node_emb, pred_feat_values = self.ge_decoder(z, E_all)

        # Graph-level label prediction
        pred_labels = self.g_label_predictor(z)

        # Node-level reconstruction loss
        predicted_value_for_loss_nonzero = torch.zeros_like(ids, dtype=pred_feat_values.dtype, device=self.device)
        predicted_value_for_loss_zero = torch.zeros(pred_feat_values.shape[0] * pred_feat_values.shape[1] - len(ids.flatten()), dtype=pred_feat_values.dtype, device=self.device)

        # For each batch, fill the mask for available nodes
        sta_nonezero = 0
        sta_zero = 0
        for i, nodes in enumerate(batch.node_names):
            n_nonezero = len(nodes)
            n_zero = pred_feat_values.shape[1] - n_nonezero
            node_indices = ids[sta_nonezero : (sta_nonezero + n_nonezero)]
            mask_ = torch.zeros(pred_feat_values.shape[1], dtype=torch.bool, device=self.device)
            mask_[node_indices] = True

            predicted_value_for_loss_nonzero[sta_nonezero : (sta_nonezero + n_nonezero)] = pred_feat_values[i, node_indices].flatten()
            predicted_value_for_loss_zero[sta_zero : (sta_zero + n_zero)] = pred_feat_values[i, ~mask_].squeeze()
            sta_nonezero += n_nonezero
            sta_zero += n_zero

        return {
            "embedding": z,
            "node_recon": recon_node_emb,
            "node_recon_for_loss_all": pred_feat_values,
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

            if self.is_regression:
                self.metrics_task_regr[f"{stage}_metrics"].update(preds, targets)
            else:
                if targets.ndim > 1 and targets.shape[1] > 1:
                    targets_class = torch.argmax(targets, dim=1)
                else:
                    targets_class = targets.long()

                self.metrics_task_class[f"{stage}_metrics"].update(preds, targets_class)

                probs = F.softmax(preds, dim=1)
                self.metrics_task_prob[f"{stage}_metrics"].update(probs, targets_class)

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
                "RMSE": MeanSquaredError(squared=False, num_outputs=self.input_dim),
                # "JSD": JensenShannonDivergence(log_prob=False),
            }
        )
        if self.is_regression:
            metrics_task_regr = MetricCollection(
                {
                    "MSE": MeanSquaredError(num_outputs=self.output_dim),
                    "MAE": MeanAbsoluteError(num_outputs=self.output_dim),
                    "PCC": PearsonCorrCoef(num_outputs=self.output_dim),
                    "RMSE": MeanSquaredError(num_outputs=self.output_dim, squared=False),
                }
            )

            self.metrics_task_regr = torch.nn.ModuleDict({f"{k}_metrics": metrics_task_regr.clone(prefix=f"{k}/label_") for k in ["train", "val", "test"]})

        else:
            metrics_task_class = MetricCollection(
                {
                    "F1_weighted": MulticlassF1Score(num_classes=self.output_dim, average="weighted"),
                    "F1_macro": MulticlassF1Score(num_classes=self.output_dim, average="macro"),
                    "F1_micro": MulticlassF1Score(num_classes=self.output_dim, average="micro"),
                    "Accuracy": MulticlassAccuracy(num_classes=self.output_dim, average="weighted"),
                    "Precision": MulticlassPrecision(num_classes=self.output_dim, average="weighted"),
                    "Recall": MulticlassRecall(num_classes=self.output_dim, average="weighted"),
                }
            )
            metrics_task_prob = MetricCollection(
                {
                    "AUROC": MulticlassAUROC(num_classes=self.output_dim, average="weighted", thresholds=None),
                }
            )

            self.metrics_task_class = torch.nn.ModuleDict({f"{k}_metrics": metrics_task_class.clone(prefix=f"{k}/label_") for k in ["train", "val", "test"]})
            self.metrics_task_prob = torch.nn.ModuleDict({f"{k}_metrics": metrics_task_prob.clone(prefix=f"{k}/label_") for k in ["train", "val", "test"]})

        # Create metrics for all stages
        self.metrics_common = torch.nn.ModuleDict({f"{k}_metrics": metrics_common.clone(prefix=k + "/recon_") for k in ["train", "val", "test"]})

    def _compute_losses(self, outputs: Dict, batch: GData, stage: str) -> Dict:
        losses = {}

        node_true_val_for_loss = batch.x.squeeze(1)

        self.metrics_common[f"{stage}_metrics"].update(outputs["node_recon_for_loss"], node_true_val_for_loss)

        # Node reconstruction loss
        recon_loss = F.mse_loss(outputs["node_recon_for_loss"], node_true_val_for_loss)

        recon_loss_zeros = F.mse_loss(
            outputs["node_recon_for_loss_zeros"],
            torch.zeros_like(outputs["node_recon_for_loss_zeros"]),
        )

        losses["recon_MSE"] = recon_loss
        losses["recon_zeros"] = recon_loss_zeros

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

        if self.current_epoch_tmp < self.current_epoch:
            self.current_epoch_tmp = self.current_epoch
            self.current_epoch_work = self.current_epoch_work + 1
            if self.current_epoch_work > 2:
                self.current_epoch_work = 0
            if self.current_epoch < 3:
                self.current_epoch_work = 0

        if self.focus_task is None:
            if stage == "train":
                if self.current_epoch_work == 0:
                    # Focus on reconstruction loss
                    total_loss = losses["recon"]

                elif self.current_epoch_work == 1:
                    # Focus on label prediction loss
                    total_loss = losses["label"] / (losses["label"] / self.loss_smooth).detach()

                else:
                    # Calculate total loss with dynamic scaling
                    total_loss = 0.5 * losses["recon"] + 0.5 * losses["label"] / (losses["label"] / self.loss_smooth).detach()

                self.loss_smooth = 0.1 * total_loss.detach() + 0.9 * self.loss_smooth

            else:
                total_loss = 0.5 * losses["recon"] + 0.5 * losses["label"] / (losses["label"] / self.loss_smooth).detach()

        elif self.focus_task == "label":
            total_loss = losses["label"]
        elif self.focus_task == "recon":
            total_loss = losses["recon"]
        else:
            raise ValueError("Invalid focus_task specified.")

        unweighted_loss = 0.5 * losses["recon"] + 0.5 * losses["label"]

        return total_loss, unweighted_loss

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
                if self.is_regression:
                    metrics_task_regr = self.metrics_task_regr[f"{stage}_metrics"].compute()
                    for name, val in metrics_task_regr.items():
                        self.log(name, val, sync_dist=True, batch_size=self._get_batch_size(stage))
                    self.metrics_task_regr[f"{stage}_metrics"].reset()
                else:
                    metrics_task_class = self.metrics_task_class[f"{stage}_metrics"].compute()
                    for name, val in metrics_task_class.items():
                        self.log(name, val, sync_dist=True, batch_size=self._get_batch_size(stage))
                    self.metrics_task_class[f"{stage}_metrics"].reset()

                    metrics_task_prob = self.metrics_task_prob[f"{stage}_metrics"].compute()
                    for name, val in metrics_task_prob.items():
                        self.log(name, val, sync_dist=True, batch_size=self._get_batch_size(stage))
                    self.metrics_task_prob[f"{stage}_metrics"].reset()

    def _get_batch_size(self, stage: str) -> int:
        if stage == "train":
            return self.trainer.train_dataloader.batch_size if self.trainer.train_dataloader is not None else 1
        elif stage == "val":
            return self.trainer.val_dataloaders.batch_size if self.trainer.val_dataloaders is not None else 1
        elif stage == "test":
            return self.trainer.test_dataloaders.batch_size if self.trainer.test_dataloaders is not None else 1
        return 1


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

    callback_es = EarlyStopping(
        monitor="val/loss_unweighted",
        patience=es_patience,
        mode="min",
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename=const.default.ckpt_fname_format,
        monitor="val/loss_unweighted",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger_tr = TensorBoardLogger(save_dir=log_dir, name="")

    trainer = Trainer(
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger_tr,
        log_every_n_steps=1,
        precision=const.default.precision,
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

    def __init__(self, args: Dict[str, Any], existing_model_path: Optional[str] = None, focus: Optional[str] = None):
        r"""
        Initialize tuning environment with parameters.

        Args:
            args (Dict[str, Any]): Dictionary containing hyperparameters and other configurations.
            existing_model_path (Optional[str]): Path to an existing model to resume training from.
            focus (Optional[str]): Focus of the task (choose from `None`, `'recon'`, `'label'`, `'recon_and_freeze'`, `'label_and_freeze'`).
        """
        self.args = args
        self.existing_model_path = existing_model_path

        if focus not in [None, "recon", "label", "recon_and_freeze", "label_and_freeze"]:
            raise ValueError("Invalid focus option. Choose from 'None', 'recon', 'label', 'recon_and_freeze', or 'label_and_freeze'.")

        self.freeze_label = False
        self.freeze_recon = False
        self.focus = focus
        if focus == "label_and_freeze":
            self.freeze_recon = True
            self.focus = "label"
        elif focus == "recon_and_freeze":
            self.freeze_label = True
            self.focus = "recon"

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
                n_workers=self.args["n_workers"],
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
            node_emb_dim = _amsgp.node_embedding_layers.embedding_dim
            fusion_dims_node_emb = _amsgp.node_embedding_layers.fusion_dims
            n_heads_node_emb = _amsgp.node_embedding_layers.n_heads
            n_heads_pooling = _amsgp.xgat_pool.num_heads
            n_hop = _amsgp.n_hop
        else:
            output_dim_g_emb = trial_params.get("output_dim_g_emb", self.args["output_dim_g_emb"])
            node_emb_dim = trial_params.get("node_emb_dim", self.args["node_emb_dim"])
            fusion_dims_node_emb = fusion_dims_node_emb
            n_heads_node_emb = trial_params.get("n_heads_node_emb", self.args["n_heads_node_emb"])
            n_heads_pooling = trial_params.get("n_heads_pooling", self.args["n_heads_pooling"])
            n_hop = trial_params.get("n_hop", self.args["n_hop"])

        _model = DeepTAN(
            z_dict_node_names=self.dict_node_names,
            input_dim=self.args["input_node_emb_dim"],
            output_g_label_dim=self.output_g_label_dim,
            is_regression=self.is_regression,
            class_weights=self.class_weight,
            node_emb_dim=node_emb_dim,
            fusion_dims_node_emb=fusion_dims_node_emb,
            output_dim_g_emb=output_dim_g_emb,
            n_hop=n_hop,
            threshold_edge_exist=trial_params.get("threshold_edge_exist", self.args["threshold_edge_exist"]),
            threshold_subgraph_overlap=trial_params.get("threshold_subgraph_overlap", self.args["threshold_subgraph_overlap"]),
            n_heads_node_emb=n_heads_node_emb,
            n_heads_pooling=n_heads_pooling,
            n_heads_ge_decoder=trial_params.get("n_heads_ge_decoder", self.args["n_heads_ge_decoder"]),
            n_heads_label_pred=trial_params.get("n_heads_label_pred", self.args["n_heads_label_pred"]),
            dropout=trial_params.get("dropout", self.args["dropout"]),
            lr=trial_params.get("lr", self.args["lr"]),
            chunk_size=self.args["chunk_size"],
            focus_task=self.focus,
            guide_gat=self.args["guide_gat"],
        )

        if self.existing_model_path is not None:
            _model.amsgp = _amsgp
            _model.ge_decoder = _ge_decoder

        if self.freeze_label:
            _model.g_label_predictor.requires_grad_(False)
            _model.ge_decoder.requires_grad_(True)
            _model.amsgp.requires_grad_(True)
        if self.freeze_recon:
            _model.g_label_predictor.requires_grad_(True)
            _model.ge_decoder.requires_grad_(False)
            _model.amsgp.requires_grad_(False)

        return _model

    def _init_model(self):
        """This function is used for create model directly without getting ``trial_params``."""

        if self.existing_model_path is not None:
            _amsgp, _ge_decoder = self._load_ckpt(self.existing_model_path, self.dict_node_names)
            output_dim_g_emb = _amsgp.output_dim_g_emb
            node_emb_dim = _amsgp.node_embedding_layers.embedding_dim
            fusion_dims_node_emb = _amsgp.node_embedding_layers.fusion_dims
            n_heads_node_emb = _amsgp.node_embedding_layers.n_heads
            n_heads_pooling = _amsgp.xgat_pool.num_heads
            n_hop = _amsgp.n_hop
        else:
            output_dim_g_emb = self.args["output_dim_g_emb"]
            node_emb_dim = self.args["node_emb_dim"]
            fusion_dims_node_emb = self.args["fusion_dims_node_emb"]
            n_heads_node_emb = self.args["n_heads_node_emb"]
            n_heads_pooling = self.args["n_heads_pooling"]
            n_hop = self.args["n_hop"]

        _model = DeepTAN(
            z_dict_node_names=self.dict_node_names,
            input_dim=self.args["input_node_emb_dim"],
            output_g_label_dim=self.output_g_label_dim,
            is_regression=self.args["is_regression"],
            class_weights=self.class_weight,
            node_emb_dim=node_emb_dim,
            fusion_dims_node_emb=fusion_dims_node_emb,
            output_dim_g_emb=output_dim_g_emb,
            n_hop=n_hop,
            threshold_edge_exist=self.args["threshold_edge_exist"],
            threshold_subgraph_overlap=self.args["threshold_subgraph_overlap"],
            n_heads_node_emb=n_heads_node_emb,
            n_heads_pooling=n_heads_pooling,
            n_heads_ge_decoder=self.args["n_heads_ge_decoder"],
            n_heads_label_pred=self.args["n_heads_label_pred"],
            dropout=self.args["dropout"],
            lr=self.args["lr"],
            chunk_size=self.args["chunk_size"],
            focus_task=self.focus,
            guide_gat=self.args["guide_gat"],
        )

        if self.existing_model_path is not None:
            _model.amsgp = _amsgp
            _model.ge_decoder = _ge_decoder

        if self.freeze_label:
            _model.g_label_predictor.requires_grad_(False)
            _model.ge_decoder.requires_grad_(True)
            _model.amsgp.requires_grad_(True)
        if self.freeze_recon:
            _model.g_label_predictor.requires_grad_(True)
            _model.ge_decoder.requires_grad_(False)
            _model.amsgp.requires_grad_(False)

        return _model

    def _load_ckpt(self, existing_model_path: str, dict_node_names_new: Dict[str, int]):
        r"""
        Load a checkpoint from the given path and extract its graph embedding and decoding module.
        """
        path_hparams = os.path.join(os.path.dirname(existing_model_path), "version_0", "hparams.yaml")
        _model_pre = DeepTAN.load_from_checkpoint(existing_model_path, map_location=get_map_location(), hparams_file=path_hparams)

        # Extract AMSGP modules
        _model_amsgp = _model_pre.amsgp
        _model_ge_decoder = _model_pre.ge_decoder

        dict_node_names_former = _model_amsgp.node_embedding_layers.dict_node_names
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
                "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.2),
                "node_emb_dim": trial.suggest_categorical("node_emb_dim", [128, 192, 256]),
                "n_heads_node_emb": trial.suggest_categorical("n_heads_node_emb", [2, 4, 8]),
                "n_heads_pooling": trial.suggest_categorical("n_heads_pooling", [2, 4, 8]),
                "n_heads_ge_decoder": trial.suggest_categorical("n_heads_ge_decoder", [2, 4, 8]),
                "n_heads_label_pred": trial.suggest_categorical("n_heads_label_pred", [2, 4, 8]),
                "fusion_dims_node_emb": trial.suggest_categorical("fusion_dims_node_emb", fusion_dims_node_emb_list_strings),
                "output_dim_g_emb": trial.suggest_categorical("output_dim_g_emb", [192, 256, 384]),
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
