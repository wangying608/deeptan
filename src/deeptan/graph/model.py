r"""
DeepTAN:
Trait-associated multi-omics network inference via multi-task NMIC-guided adaptive multi-scale graph embedding.
"""

from typing import List, Dict, Optional, Any
from sympy import flatten
import pickle
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
import lightning as ltn
from lightning import Trainer, LightningDataModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from litdata import StreamingDataset, StreamingDataLoader
from torch_geometric.data import Data as GData
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
)
import deeptan.constants as const
from deeptan.graph.modules import AMSGP, GE_Decoder, GLabelPredictor
from deeptan.utils.uni import collate_fn, get_map_location

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
        class_weights: Optional[List[float]] = None,
        node_emb_dim: int = 128,
        fusion_dims_node_emb: List[int] = [256, 512, 128],
        output_dim_g_emb: int = 512,
        n_hop: int = 2,
        threshold_edge_exist: float = 0.1,
        threshold_subgraph_overlap: float = 0.99,
        n_heads_node_emb: int = 4,
        n_heads_pooling: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-4,
        negative_slope: float = 0.1,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dict_node_names = dict_node_names

        self.output_dim = output_g_label_dim if output_g_label_dim is not None else 2
        # self.class_weights = (
        #     torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        #     if class_weights is not None
        #     else None
        # )
        self.class_weights = class_weights

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
            dropout=dropout,
            negative_slope=negative_slope,
        )

        # Graph-level label predictor
        self.g_label_predictor = GLabelPredictor(
            output_dim_g_emb, self.output_dim, [512, 512, 256], dropout
        )

        # Metrics and initialization
        self._init_metrics()
        self.ema_loss = None

    def forward(self, batch: GData) -> Dict[str, Any]:
        # Check input dimension of x
        # print(f"\nInput x shape: {batch.x.shape}")
        # # Check data distribution of x
        # print(f"Input x distribution: mean = {batch.x.mean()}, std = {batch.x.std()}")
        # # Check if x has the correct number of dimensions
        # print(f"Input x dimensions: {batch.x.dim()}\n")
        assert batch.x is not None, "Input x is None"
        assert batch.edge_index is not None, "Input edge_index is None"
        assert batch.x.dim() == 2, f"The input dim is wrong: {batch.x.shape}"
        assert batch.edge_index.max() < batch.x.size(0), (
            f"The edge index is wrong: {batch.edge_index.shape}"
        )

        # Check if all node names are valid
        for nodes in batch.node_names:
            assert all(n in self.dict_node_names for n in nodes), (
                f"Node names are not valid: {batch.node_names}"
            )

        # Extract batch information if available, otherwise initialize with zeros
        node_batch = getattr(
            batch,
            "batch",
            torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device),
        )
        # print(f"Batch information: {node_batch}\n")

        # Feature extraction
        z, E_i, E_all = self.amsgp(
            node_names=batch.node_names,
            x=batch.x,
            edge_attr=batch.edge_attr,
            edge_index=batch.edge_index,
            batch=node_batch,
        )

        recon_node_emb, recon_node_val_for_loss_all = self.ge_decoder(z, E_i, E_all)

        # print(f"Reconstructed node embeddings: {recon_node_emb.shape}")
        # print(f"Reconstructed node values for loss: {recon_node_val_for_loss_all.shape}")

        # Graph-level label prediction
        pred_labels = self.g_label_predictor(z)
        # if not self.hparams.is_regression:
        #     pred_labels = F.softmax(pred_labels, dim=1)

        # Node-level reconstruction loss
        recon_node_val_for_loss_list = [
            recon_node_val_for_loss_all[
                i, self.pick_avail_node_in_x(batch.node_names[i]), :
            ]
            for i in range(len(batch.node_names))
        ]
        recon_node_val_for_loss = torch.cat(recon_node_val_for_loss_list)

        # Node-level reconstruction loss for zeros
        recon_node_val_for_loss_list = [
            recon_node_val_for_loss_all[
                i, self.pick_unavail_node_in_x(batch.node_names[i]), :
            ]
            for i in range(len(batch.node_names))
        ]
        recon_node_val_for_loss_zeros = torch.cat(recon_node_val_for_loss_list)

        return {
            "embedding": z,
            "node_recon": recon_node_emb,
            "label_pred": pred_labels,
            "node_recon_for_loss": recon_node_val_for_loss,
            "node_recon_for_loss_zeros": recon_node_val_for_loss_zeros,
            "node_recon_for_loss_all": recon_node_val_for_loss_all,
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
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

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
        assert batch.x is not None
        losses = {}

        node_recon_for_loss = outputs["node_recon_for_loss"].squeeze(1)
        node_true_val_for_loss = batch.x.squeeze(1)

        self.metrics_common[f"{stage}_metrics"].update(
            node_recon_for_loss, node_true_val_for_loss
        )

        # Node reconstruction loss
        recon_loss = F.mse_loss(node_recon_for_loss, node_true_val_for_loss)

        kl_loss = F.kl_div(
            F.log_softmax(node_recon_for_loss, dim=1),
            F.softmax(node_true_val_for_loss, dim=1),
            log_target=True,
            reduction="mean",
        )

        recon_loss_zeros = F.mse_loss(
            outputs["node_recon_for_loss_zeros"].squeeze(1),
            torch.zeros_like(outputs["node_recon_for_loss_zeros"].squeeze(1)),
        )

        losses["recon_KLD"] = kl_loss
        losses["recon_MSE"] = recon_loss
        losses["recon_zeros"] = recon_loss_zeros

        # Total reconstruction loss
        losses["recon"] = recon_loss + kl_loss + recon_loss_zeros
        # losses["recon"] = recon_loss  # + 0.2 * recon_loss_zeros

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

                if self.class_weights is None:
                    pred_loss = F.cross_entropy(outputs["label_pred"], _y)
                else:
                    pred_loss = F.cross_entropy(
                        outputs["label_pred"],
                        _y,
                        weight=torch.tensor(
                            self.class_weights, dtype=torch.float32, device=self.device
                        ),
                    )

            losses["label"] = pred_loss

        # Dynamic weight adjustment
        total_loss = self._balance_losses(losses, stage)
        return {**losses, "loss": total_loss}

    def _balance_losses(self, losses: Dict, stage: str) -> torch.Tensor:
        # Dynamic weight adjustment
        if stage == "train" and self.current_epoch > 0:
            with torch.no_grad():
                loss_values = torch.stack([losses[k] for k in ["label", "recon"]])
                mean_loss = loss_values.mean() + 1e-8
                task_weights = F.softmax(loss_values / mean_loss, dim=0)
                self.hparams.alpha = 0.8 * task_weights[0] + 0.2 * self.hparams.alpha

        # EMA stableization
        # total = (
        #     self.hparams.alpha * losses["label"]
        #     + (1 - self.hparams.alpha) * losses["recon"]
        # )

        # if self.ema_loss is None:
        #     self.ema_loss = total.detach()
        # else:
        #     self.ema_loss = 0.9 * self.ema_loss + 0.1 * total.detach()

        # return total + 0.1 * (total - self.ema_loss).abs()

        total_loss = (
            self.hparams.alpha * losses["label"]
            + (1 - self.hparams.alpha) * losses["recon"]
        )
        return total_loss
        # return losses["label"]

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


def train_model(
    model: Any,
    datamodule: LightningDataModule,
    es_patience: int,
    max_epochs: int,
    min_epochs: int,
    log_dir: str,
    accumulate_grad_batches: int = 4,
    # devices: Union[list[int], str, int] = const.default.devices,
    accelerator: str = const.default.accelerator,
    fast_dev_run: bool = False,
):
    r"""Fit the model."""
    # avail_dev = get_avail_nvgpu(devices)

    torch.autograd.set_detect_anomaly(True)

    callback_es = EarlyStopping(
        monitor=const.dkey.title_val_loss,
        patience=es_patience,
        mode="min",
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename=const.default.ckpt_fname_format,
        monitor=const.dkey.title_val_loss,
    )

    logger_tr = TensorBoardLogger(save_dir=log_dir, name="")

    trainer = Trainer(
        fast_dev_run=fast_dev_run,
        # strategy="ddp_spawn",
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger_tr,
        log_every_n_steps=1,
        precision="16-mixed",
        # devices=avail_dev,
        accelerator=accelerator,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[callback_es, callback_ckpt, StochasticWeightAveraging(swa_lrs=1e-4)],
        num_sanity_val_steps=0,
        default_root_dir=log_dir,
        gradient_clip_val=1.0,
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


def predict(
    model_ckpt_path: str,
    litdata_dir: str,
    output_pickle_path: str,
    map_location: Optional[str] = None,
    batch_size: int = 1,
):
    # Load a DeepTAN model
    model = DeepTAN.load_from_checkpoint(
        model_ckpt_path, map_location=get_map_location(map_location)
    )
    # Freeze the model
    model.eval()
    model.freeze()

    # Load the LitData dataset
    dataloader = StreamingDataLoader(
        StreamingDataset(litdata_dir), batch_size=batch_size, collate_fn=collate_fn
    )

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
