r"""
Universal functions.
"""

import time
import random
import string
from typing import Optional, Any, List
import pickle
import numpy as np
import polars as pl
from lightning import Trainer, LightningDataModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from litdata import StreamingDataset, StreamingDataLoader
import torch
from torch.cuda import device_count
from torch_geometric.data import Batch
from multiprocessing import cpu_count
import deeptan.constants as const
from deeptan.graph.model import DeepTAN


def collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


def get_avail_cpu_count(target_n: int) -> int:
    total_n = cpu_count()
    n_cpu = target_n
    if target_n <= 0:
        n_cpu = total_n
    else:
        n_cpu = min(target_n, total_n)
    return n_cpu


def get_map_location(map_loc: Optional[str] = None):
    if map_loc is None:
        if device_count() > 0:
            which_dev = find_usable_cuda_devices(1)
            if len(which_dev) == 0:
                return "cpu"
            else:
                return f"cuda:{which_dev[0]}"
        else:
            return "cpu"
    else:
        return map_loc


def time_string() -> str:
    _time_str = time.strftime(const.default.time_format, time.localtime())
    return _time_str


def random_string(length: int = 7) -> str:
    letters = string.ascii_letters + string.digits
    result = "".join(random.choice(letters) for _ in range(length))
    return result


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
