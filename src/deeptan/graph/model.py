r"""
DeepTAN:
Trait-associated multi-omics network inference via multi-task NMIC-guided adaptive multi-scale graph embedding.
"""

from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
import lightning as ltn
from torch_geometric.data import Data as GData

# from torch_geometric.utils import unbatch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
    MatthewsCorrCoef,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)
import deeptan.constants as const
from deeptan.graph.core import AMSGP
from deeptan.graph.modules import GE_Decoder, GLabelPredictor, EdgeDecoder

torch.set_float32_matmul_precision(const.default.matmul_precision)


class AMSGPMTL(ltn.LightningModule):
    r"""
    AMSGP for semi-supervised multi-task learning with enhanced training strategies.
    """

    def __init__(
        self,
        dict_node_names: Dict[str, int],
        input_dim: int,
        output_dim: int,
        is_regression: bool,
        node_emb_dim: int = 128,
        fusion_dims_node_emb: List[int] = [256, 128],
        output_dim_g_emb: int = 128,
        n_hop: int = 3,
        threshold_edge_exist: float = 0.5,
        threshold_subgraph_overlap: float = 0.6,
        n_heads_node_emb: int = 4,
        n_heads_pooling: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-3,
        negative_slope: float = 0.2,
        alpha: float = 0.7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Core components
        self.amsgp = AMSGP(
            dict_node_names=dict_node_names,
            input_dim=input_dim,
            node_emb_dim=node_emb_dim,
            fusion_dims_node_emb=fusion_dims_node_emb,
            n_heads_node_emb=n_heads_node_emb,
            output_dim_g_emb=output_dim_g_emb,
            n_heads_pooling=n_heads_pooling,
            n_hop=n_hop,
            threshold_edge_exist=threshold_edge_exist,
            threshold_subgraph_overlap=threshold_subgraph_overlap,
            negative_slope=negative_slope,
        )

        # Multi-task decoders
        self.ge_decoder = GE_Decoder(
            z_dim=output_dim_g_emb,
            h_dim=node_emb_dim,
            output_dim=fusion_dims_node_emb[-1],
            hidden_dim=256,
        )
        self.edge_recon = EdgeDecoder(fusion_dims_node_emb[-1])
        self.g_label_predictor = GLabelPredictor(
            output_dim_g_emb, output_dim, [512, 256]
        )

        # Metrics and initialization
        # if not hasattr(self, "metrics"):
        self._init_metrics()
        self.ema_loss = None

    def forward(self, g: GData) -> Dict[str, Any]:
        # Extract batch information if available, otherwise initialize with zeros
        node_batch = getattr(
            g, "batch", torch.zeros(g.x.size(0), dtype=torch.long, device=g.x.device)
        )

        # Feature extraction
        z, Embedding = self.amsgp(
            node_names=g.node_names,
            x=g.x,
            edge_attr=g.edge_attr,
            edge_index=g.edge_index,
            batch=node_batch,
        )

        # print(f"\n\nz shape: {z.shape}")

        recon_node_emb = self.ge_decoder(z, Embedding)
        # batch_size = recon_node_emb.size(0)
        # print(f"\nReconstructed node embeddings shape: {recon_node_emb.shape}\n")  # torch.Size([16, 50, 32])

        edge_pred = self.edge_recon(recon_node_emb, g.edge_index)

        # Graph-level label prediction
        pred_labels = self.g_label_predictor(z)
        if not self.hparams.is_regression:
            pred_labels = torch.nn.functional.log_softmax(pred_labels, dim=1)

        # Assertions to check the dimensions of the outputs
        # assert z.dim() == 2 and z.size(0) == g.batch.max() + 1, (
        #     f"图嵌入形状错误，应为[batch_size, dim]，实际得到{z.shape}"
        # )
        # assert pred_labels.size(0) == z.size(0), "预测结果数量与图数量不匹配"

        return {
            "embedding": z,
            "node_recon": recon_node_emb.view(-1, self.hparams.input_dim),
            "label_pred": pred_labels,
            "edge_recon": edge_pred,
        }

        # print("\n\nGraph embedding shape:", z.shape)
        # print("Reconstructed node embedding shape:", recon_node_emb.shape)
        # print("Predicted label shape:", predicted_label.shape)
        # print("Reconstructed edge shape:", recon_edge.shape, "\n\n")

    def training_step(self, batch: GData, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: GData, batch_idx: int) -> Optional[torch.Tensor]:
        return self._shared_step(batch, "val")

    def test_step(self, batch: GData, batch_idx: int) -> Optional[torch.Tensor]:
        return self._shared_step(batch, "test")

    def predict_step(self, batch: GData, batch_idx: int) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self(batch)

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4, betas=(0.9, 0.98)
        )

        # Combined scheduler
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * 0.1)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[
                        torch.optim.lr_scheduler.LambdaLR(
                            opt, lr_lambda=lambda step: min(step / warmup_steps, 1.0)
                        ),
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt,
                            T_max=total_steps - warmup_steps,
                            eta_min=self.hparams.lr / 100,
                        ),
                    ],
                    milestones=[warmup_steps],
                ),
                "interval": "step",
                "frequency": 1,
            },
        }

    def _init_metrics(self):
        if hasattr(self, "metrics"):
            self.metrics.clear()

        # Common metrics for both tasks
        common_metrics = {
            "recon": MeanSquaredError(),
            "edge_recon": MeanSquaredError(),
        }

        # Task-specific metrics
        if self.hparams.is_regression:
            task_metrics = {
                "label_mse": MeanSquaredError(),
                "label_mae": MeanAbsoluteError(),
                "label_rmse": MeanSquaredError(squared=False),
                "label_r2": R2Score(),
                "label_pcc": PearsonCorrCoef(),
            }
        else:
            task_metrics = {
                # "label_acc": MulticlassAccuracy(
                #     num_classes=self.hparams.output_dim, average="weighted"
                # ),
                "label_f1_weighted": MulticlassF1Score(
                    num_classes=self.hparams.output_dim, average="weighted"
                ),
                "label_f1_macro": MulticlassF1Score(
                    num_classes=self.hparams.output_dim,
                    average="macro",
                ),
                "label_f1_micro": MulticlassF1Score(
                    num_classes=self.hparams.output_dim,
                    average="micro",
                ),
                "label_auc": MulticlassAUROC(
                    num_classes=self.hparams.output_dim, average="macro"
                ),
                # "label_precision": MulticlassPrecision(
                #     num_classes=self.hparams.output_dim, average="weighted"
                # ),
                # "label_recall": MulticlassRecall(
                #     num_classes=self.hparams.output_dim, average="weighted"
                # ),
                # "label_mcc": MatthewsCorrCoef(
                #     task="multiclass", num_classes=self.hparams.output_dim
                # ),
            }

        metrics_common = MetricCollection({**common_metrics})
        metrics_task_label = MetricCollection({**task_metrics})

        # Create metrics for all stages
        self.metrics_common = torch.nn.ModuleDict(
            {
                f"{k}_metrics": metrics_common.clone(prefix=k + "_")
                for k in ["train", "val", "test"]
            }
        )
        self.metrics_task_label = torch.nn.ModuleDict(
            {
                f"{k}_metrics": metrics_task_label.clone(prefix=k + "_")
                for k in ["train", "val", "test"]
            }
        )

    def _shared_step(self, batch: GData, stage: str) -> torch.Tensor:
        outputs = self(batch)
        losses = self._compute_losses(outputs, batch, stage)

        # More loss computation
        if batch.y is not None:
            preds = outputs["label_pred"]
            targets = batch.y
            targets = torch.as_tensor(targets, device=preds.device, dtype=torch.long)

            self.metrics_task_label[f"{stage}_metrics"].update(preds, targets)

        # Optimization step (only during training)
        if stage == "train":
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(losses["total"])
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()

        # Logging metrics and losses
        self._log_metrics(losses, stage)
        return losses["total"]

    def _compute_losses(self, outputs: Dict, batch: GData, stage: str) -> Dict:
        losses = {}

        # Node reconstruction loss
        recon_loss = F.mse_loss(outputs["node_recon"], batch.x)
        kl_loss = F.kl_div(
            F.log_softmax(outputs["node_recon"], dim=-1),
            F.softmax(batch.x, dim=-1),
            reduction="batchmean",
        )
        losses["recon"] = recon_loss + kl_loss

        # Edge reconstruction loss
        edge_mask = (batch.edge_attr > self.hparams.threshold_edge_exist).float()
        edge_loss = F.binary_cross_entropy(outputs["edge_recon"], edge_mask)
        losses["edge"] = edge_loss

        # Graph-level label prediction loss
        if batch.y is not None:
            if self.hparams.is_regression:
                pred_loss = F.mse_loss(outputs["label_pred"], batch.y)
            else:
                pred_loss = F.cross_entropy(outputs["label_pred"], batch.y)
            losses["pred"] = pred_loss

        # Dynamic weight adjustment
        total_loss = self._balance_losses(losses, stage)
        return {**losses, "total": total_loss}

    def _balance_losses(self, losses: Dict, stage: str) -> torch.Tensor:
        # Dynamic weight adjustment
        if stage == "train" and self.current_epoch > 5:
            with torch.no_grad():
                loss_values = torch.stack(
                    [losses[k] for k in ["pred", "recon", "edge"]]
                )
                task_weights = F.softmax(loss_values / loss_values.mean(), dim=0)
                self.hparams.alpha = 0.8 * task_weights[0] + 0.2 * self.hparams.alpha

        # EMA stableization
        total = self.hparams.alpha * losses.get("pred", 0) + (
            1 - self.hparams.alpha
        ) * (losses["recon"] + losses["edge"])

        if self.ema_loss is None:
            self.ema_loss = total.detach()
        else:
            self.ema_loss = 0.9 * self.ema_loss + 0.1 * total.detach()

        return total + 0.1 * (total - self.ema_loss).abs()

    def _log_metrics(self, losses: Dict, stage: str):
        # Log losses
        for k, v in losses.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=(k == "total"),
                sync_dist=True,
                batch_size=self._get_batch_size(stage),
            )

        # Log evaluation metrics
        if not self.trainer.sanity_checking:
            metrics_common = self.metrics_common[f"{stage}_metrics"].compute()
            for name, val in metrics_common.items():
                self.log(
                    # f"{stage}_{name}",
                    name,
                    val,
                    sync_dist=True,
                    batch_size=self._get_batch_size(stage),
                )
            self.metrics_common[f"{stage}_metrics"].reset()

            metrics_task_label = self.metrics_task_label[f"{stage}_metrics"].compute()
            for name, val in metrics_task_label.items():
                self.log(
                    # f"{stage}_{name}",
                    name,
                    val,
                    sync_dist=True,
                    batch_size=self._get_batch_size(stage),
                )
            self.metrics_task_label[f"{stage}_metrics"].reset()

    def _get_batch_size(self, stage: str) -> int:
        if stage == "train":
            return self.trainer.train_dataloader.batch_size
        elif stage == "val":
            return self.trainer.val_dataloaders.batch_size
        elif stage == "test":
            return self.trainer.test_dataloaders.batch_size
        return 1

    # Define epoch-end handlers
    # def on_train_epoch_end(self):
    #     self._log_epoch_metrics("train")

    # def on_validation_epoch_end(self):
    #     self._log_epoch_metrics("val")

    # def on_test_epoch_end(self):
    #     self._log_epoch_metrics("test")

    # def _log_epoch_metrics(self, stage: str):
    #     metrics = self.metrics_common[f"{stage}_metrics"].compute()
    #     for name, val in metrics.items():
    #         self.log(f"{stage}_{name}", val, sync_dist=True)
    #     self.metrics_common[f"{stage}_metrics"].reset()

    #     metrics = self.metrics_task_label[f"{stage}_metrics"].compute()
    #     for name, val in metrics.items():
    #         self.log(f"{stage}_{name}", val, sync_dist=True)
    #     self.metrics_task_label[f"{stage}_metrics"].reset()
