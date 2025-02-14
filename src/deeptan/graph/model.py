r"""
DeepTAN:
Trait-associated multi-omics network inference via multi-task NMIC-guided adaptive multi-scale graph embedding.
"""

from typing import List, Dict, Optional, Any
from sympy import flatten
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
import lightning as ltn
from torch_geometric.data import Data as GData
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
    # MatthewsCorrCoef,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
)
import deeptan.constants as const
from deeptan.graph.core import AMSGP
from deeptan.graph.modules import GE_Decoder, GLabelPredictor

torch.set_float32_matmul_precision(const.default.matmul_precision)


class DeepTAN(ltn.LightningModule):
    r"""
    DeepTAN.
    """

    def __init__(
        self,
        dict_node_names: Dict[str, int],
        input_dim: int,
        output_g_label_dim: Optional[int],
        is_regression: bool,
        node_emb_dim: int = 128,
        fusion_dims_node_emb: List[int] = [256, 512, 128],
        output_dim_g_emb: int = 512,
        n_hop: int = 2,
        threshold_edge_exist: float = 0.5,
        threshold_subgraph_overlap: float = 0.6,
        n_heads_node_emb: int = 4,
        n_heads_pooling: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-4,
        negative_slope: float = 0.2,
        alpha: float = 0.7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.dict_node_names = dict_node_names

        if output_g_label_dim is None:
            self.output_dim = 2
        else:
            self.output_dim = output_g_label_dim

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
            dropout=dropout,
        )

        # Multi-task decoders
        self.ge_decoder = GE_Decoder(
            z_dim=output_dim_g_emb,
            h_dim=node_emb_dim,
            output_dim=input_dim,
            hidden_dim=512,
        )

        # Graph-level label predictor
        self.g_label_predictor = GLabelPredictor(
            output_dim_g_emb, self.output_dim, [512, 512, 256], dropout
        )

        # Metrics and initialization
        self._init_metrics()
        self.ema_loss = None

    def forward(self, g: GData) -> Dict[str, Any]:
        assert g.x.dim() == 2, f"The input dim is wrong: {g.x.shape}"
        assert g.edge_index.max() < g.x.size(0), (
            "The edge index is wrong: {g.edge_index.shape}"
        )
        # Check if all node names are valid
        for nodes in g.node_names:
            assert all(n in self.dict_node_names for n in nodes), (
                "Node names are not valid: {g.node_names}"
            )
        # Extract batch information if available, otherwise initialize with zeros
        node_batch = getattr(
            g, "batch", torch.zeros(g.x.size(0), dtype=torch.long, device=g.x.device)
        )

        # Feature extraction
        z = self.amsgp(
            node_names=g.node_names,
            x=g.x,
            edge_attr=g.edge_attr,
            edge_index=g.edge_index,
            batch=node_batch,
        )

        recon_node_emb, recon_node_emb_for_loss_all = self.ge_decoder(
            z, self.amsgp.node_embedding_layers.embed
        )

        # Graph-level label prediction
        pred_labels = self.g_label_predictor(z)
        if not self.hparams.is_regression:
            pred_labels = F.softmax(pred_labels, dim=1)

        # Node-level reconstruction loss
        recon_node_emb_for_loss_list = [
            recon_node_emb_for_loss_all[
                i, self.pick_avail_node_in_x(g.node_names[i]), :
            ].contiguous()
            for i in range(len(g.node_names))
        ]
        recon_node_emb_for_loss = torch.cat(recon_node_emb_for_loss_list)

        # Node-level reconstruction loss for zeros
        # recon_node_emb_for_loss_list = [
        #     recon_node_emb_for_loss_all[
        #         i, self.pick_unavail_node_in_x(g.node_names[i]), :
        #     ].contiguous()
        #     for i in range(len(g.node_names))
        # ]
        # recon_node_emb_for_loss_zeros = torch.cat(recon_node_emb_for_loss_list)

        return {
            "embedding": z,
            "node_recon": recon_node_emb,
            "label_pred": pred_labels,
            "node_recon_for_loss": recon_node_emb_for_loss,
            # "node_recon_for_loss_zeros": recon_node_emb_for_loss_zeros,
            "node_recon_for_loss_all": recon_node_emb_for_loss_all,
        }

        # print("\n\nGraph embedding shape:", z.shape)
        # print("Reconstructed node embedding shape:", recon_node_emb.shape)
        # print("Predicted label shape:", predicted_label.shape)

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
            # print(f"\npreds:\n{preds.shape}\ntargets:{targets.shape}\n")
            if not self.hparams.is_regression:
                if targets.ndim > 1 and targets.shape[1] > 1:
                    targets = torch.argmax(targets, dim=1)
            self.metrics_task_label[f"{stage}_metrics"].update(preds, targets)

        # Logging metrics and losses
        self._log_metrics(losses, stage)
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4, betas=(0.9, 0.98)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 10, 2
        )
        # scaler = torch.amp.GradScaler()
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
            # "scaler": scaler,
        }

        # # Combined scheduler
        # total_steps = int(self.trainer.estimated_stepping_batches)
        # total_steps = max(total_steps, 1)
        # warmup_steps = max(500, int(total_steps * 0.2))
        # T_max = max(1, total_steps - warmup_steps)

        # # Define the warmup scheduler
        # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     opt, lr_lambda=lambda step: min(step / warmup_steps, 1.0)
        # )

        # # Define the cosine annealing scheduler
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt, T_max=T_max, eta_min=self.hparams.lr / 100
        # )

        # # Combine the schedulers
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     opt,
        #     schedulers=[warmup_scheduler, cosine_scheduler],
        #     milestones=[warmup_steps],
        # )

        # return {
        #     "optimizer": opt,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "frequency": 1,
        #     },
        # }

    # def on_after_backward(self):
    #     grad_norm = 0.0
    #     for p in self.parameters():
    #         if p.grad is not None:
    #             grad_norm += p.grad.data.norm(2).item() ** 2
    #     self.log("grad_norm", grad_norm**0.5, prog_bar=True)

    #     if grad_norm > 5e4:
    #         for param_group in self.optimizers()._optimizer.param_groups:
    #             param_group["lr"] *= 0.5
    #             self.log("lr", param_group["lr"], prog_bar=True)

    def _init_metrics(self):
        if hasattr(self, "metrics"):
            self.metrics.clear()

        metrics_common = MetricCollection(
            {
                "MSE": MeanSquaredError(num_outputs=self.hparams.input_dim),
                "MAE": MeanAbsoluteError(num_outputs=self.hparams.input_dim),
                "PCC": PearsonCorrCoef(num_outputs=self.hparams.input_dim),
            }
        )
        if self.hparams.is_regression:
            metrics_task_label = MetricCollection(
                {
                    "MSE": MeanSquaredError(num_outputs=self.output_dim),
                    "MAE": MeanAbsoluteError(num_outputs=self.output_dim),
                    "PCC": PearsonCorrCoef(num_outputs=self.output_dim),
                    "RMSE": MeanSquaredError(
                        num_outputs=self.output_dim, squared=False
                    ),
                }
            )
        else:
            metrics_task_label = MetricCollection(
                {
                    "F1_weighted": MulticlassF1Score(
                        num_classes=self.output_dim, average="weighted"
                    ),
                    "F1_macro": MulticlassF1Score(
                        num_classes=self.output_dim, average="macro"
                    ),
                    "F1_micro": MulticlassF1Score(
                        num_classes=self.output_dim, average="micro"
                    ),
                    "Accuracy": MulticlassAccuracy(num_classes=self.output_dim),
                    "Precision": MulticlassPrecision(
                        num_classes=self.output_dim, average="weighted"
                    ),
                    "Recall": MulticlassRecall(
                        num_classes=self.output_dim, average="weighted"
                    ),
                    "AUROC": MulticlassAUROC(
                        num_classes=self.output_dim, average="macro"
                    ),
                    # "MCC": MatthewsCorrCoef(
                    #     task="multiclass", num_classes=self.output_dim
                    # ),
                }
            )

        # Create metrics for all stages
        self.metrics_common = torch.nn.ModuleDict(
            {
                f"{k}_metrics": metrics_common.clone(prefix=k + "_recon_")
                for k in ["train", "val", "test"]
            }
        )
        self.metrics_task_label = torch.nn.ModuleDict(
            {
                f"{k}_metrics": metrics_task_label.clone(prefix=k + "_label_")
                for k in ["train", "val", "test"]
            }
        )

    def _compute_losses(self, outputs: Dict, batch: GData, stage: str) -> Dict:
        losses = {}

        self.metrics_common[f"{stage}_metrics"].update(
            outputs["node_recon_for_loss"], batch.x
        )

        # Node reconstruction loss
        recon_loss = F.mse_loss(outputs["node_recon_for_loss"], batch.x)
        # kl_loss = F.kl_div(
        #     F.log_softmax(outputs["node_recon_for_loss"], dim=-1),
        #     F.softmax(batch.x, dim=-1),
        #     reduction="batchmean",
        # )
        # recon_loss_zeros = F.mse_loss(
        #     outputs["node_recon_for_loss_zeros"],
        #     torch.zeros_like(outputs["node_recon_for_loss_zeros"]),
        # )
        # losses["recon_KLD"] = kl_loss
        losses["recon_MSE"] = recon_loss
        # losses["recon_zeros"] = recon_loss_zeros
        # losses["recon"] = recon_loss + kl_loss + 0.2 * recon_loss_zeros
        losses["recon"] = recon_loss# + 0.2 * recon_loss_zeros

        # Graph-level label prediction loss
        if batch.y is None:
            # Placeholder for no label loss
            losses["label"] = torch.tensor(0.0, device=self.device)
        else:
            if isinstance(batch.y, torch.Tensor):
                _y = batch.y
            else:
                _y = torch.tensor(batch.y, device=self.device)
            if self.hparams.is_regression:
                pred_loss = F.mse_loss(outputs["label_pred"], _y)
            else:
                if _y.ndim > 1 and _y.shape[1] > 1:
                    _y = torch.argmax(_y, dim=1)
                pred_loss = F.cross_entropy(outputs["label_pred"], _y)
            losses["label"] = pred_loss

        # Dynamic weight adjustment
        total_loss = self._balance_losses(losses, stage)
        return {**losses, "loss": total_loss}

    def _balance_losses(self, losses: Dict, stage: str) -> torch.Tensor:
        # Dynamic weight adjustment
        if stage == "train":  # and self.current_epoch > 5:
            with torch.no_grad():
                loss_values = torch.stack([losses[k] for k in ["label", "recon"]])
                task_weights = F.softmax(loss_values / loss_values.mean(), dim=0)
                self.hparams.alpha = 0.8 * task_weights[0] + 0.2 * self.hparams.alpha

        # EMA stableization
        total = (
            self.hparams.alpha * losses["label"]
            + (1 - self.hparams.alpha) * losses["recon"]
        )

        if self.ema_loss is None:
            self.ema_loss = total.detach()
        else:
            self.ema_loss = 0.9 * self.ema_loss + 0.1 * total.detach()

        return total + 0.1 * (total - self.ema_loss).abs()

        # alpha = 0.5
        # total = alpha * losses["label"] + (1 - alpha) * losses["recon"]
        # return total

    def _log_metrics(self, losses: Dict, stage: str):
        for k, v in losses.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=(k == "loss"),
                sync_dist=True,
                batch_size=self._get_batch_size(stage),
            )

        # Log evaluation metrics
        if not self.trainer.sanity_checking:
            metrics_common = self.metrics_common[f"{stage}_metrics"].compute()
            for name, val in metrics_common.items():
                self.log(
                    name,
                    val,
                    sync_dist=True,
                    batch_size=self._get_batch_size(stage),
                )
            self.metrics_common[f"{stage}_metrics"].reset()

            if self.hparams.output_g_label_dim is not None:
                metrics_task_label = self.metrics_task_label[
                    f"{stage}_metrics"
                ].compute()
                for name, val in metrics_task_label.items():
                    self.log(
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

    def pick_avail_node_in_x(self, x_node_names: List[str]):
        r"""
        Pick available features (nodes) from self.hparams.dict_node_names in x_node_names.
        Returns:
            List[str]: A list of available node indices (the indices in self.hparams.dict_node_names).
        """
        # Extract all node names from x_node_names, even if they are nested lists or tuples.
        node_names = flatten(x_node_names)
        avail_node_ind = [self.dict_node_names[node] for node in node_names]
        return avail_node_ind

        # Also return the node that are not available.
        # unavail_node_ind = [
        #     i for i in range(len(self.dict_node_names)) if i not in avail_node_ind
        # ]
        # return avail_node_ind, unavail_node_ind

    def pick_unavail_node_in_x(self, x_node_names: List[str]):
        r"""
        Pick unavailable features (nodes) from self.hparams.dict_node_names in x_node_names.
        Returns:
            List[str]: A list of unavailable node indices (the indices in self.hparams.dict_node_names).
        """
        # Extract all node names from x_node_names, even if they are nested lists or tuples.
        node_names = flatten(x_node_names)
        unavail_node_ind: List[int] = []
        for node in self.dict_node_names:
            if node not in node_names:
                unavail_node_ind.append(self.dict_node_names[node])
        return unavail_node_ind
