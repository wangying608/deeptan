r"""
Universal functions.
"""

# import os
import time
import random
# import shortuuid
import string
from typing import Optional, Union, List, Dict, Any
# import numpy as np
# import polars as pl
from lightning import Trainer, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from multiprocessing import cpu_count
import deeptan.constants as const


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
        callbacks=[callback_es, callback_ckpt],
        # num_sanity_val_steps=0,
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
