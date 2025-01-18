r"""
Universal functions.
"""
import os
import time
import random
import shortuuid
import string
from typing import Optional, Union, List, Dict, Any
import numpy as np
import polars as pl
from litdata import StreamingDataLoader, StreamingDataset
from torch import Tensor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
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

def get_avail_nvgpu(devices: Union[list[int], str, int] = const.default.devices):
    if type(devices) == int and device_count() > 0:
        avail_dev = find_usable_cuda_devices(devices)
    elif devices == 'auto' and device_count() > 0:
        avail_dev = find_usable_cuda_devices()
    else:
        avail_dev = devices
    return avail_dev

def get_map_location(map_loc: Optional[str] = None):
    if map_loc is None:
        if device_count() > 0:
            which_dev = find_usable_cuda_devices(1)
            if len(which_dev) == 0:
                return 'cpu'
            else:
                return f'cuda:{which_dev[0]}'
        else:
            return 'cpu'
    else:
        return map_loc


def time_string() -> str:
    _time_str = time.strftime(const.default.time_format, time.localtime())
    return _time_str

def random_string(length: int = 7) -> str:
    letters = string.ascii_letters + string.digits
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


# def auto_proc_feat4trn(in_array: np.ndarray, threshold_ptp: float=100.0):
#     r""" Auto preprocessing for features in training data.

#     Args:
#         in_array: Input array with shape ``(n_features, n_samples)``.
        
#         threshold_ptp: Threshold for the range.

#     Steps:
#         1. Remove feature if its range is similar to zero.
#         2. Apply scale and log2 transformation to each feature in the training set. (Apply log2 transformation if the range of the array is larger than the threshold.)
    
#     """
#     values_min = np.min(in_array, axis=1)
#     values_max = np.max(in_array, axis=1)
#     values_ptp = values_max - values_min

#     out_array = np.copy(in_array) + 1e-4
#     feat2rm = np.where(values_ptp < 1e-3)[0]
    
#     for i in range(in_array.shape[0]):
#         if i in feat2rm:
#             continue
#         if values_ptp[i] > threshold_ptp:
#             # Apply log2 transformation
#             out_array[i, :] = np.log2(out_array[i, :])
    
#     out_array = np.delete(out_array, feat2rm, axis=0)

#     values_min = np.min(out_array, axis=1)
#     values_max = np.max(out_array, axis=1)
#     # out_array = (out_array - values_min) / (values_max - values_min)
#     out_array = (out_array - values_min[:, None]) / (values_max[:, None] - values_min[:, None])
#     return out_array, values_min, values_max, feat2rm


def omics_tensor_list_to_np(batch: List[Tensor]):
    concatenated = np.concatenate([ts.numpy() for ts in batch], axis=None)
    return concatenated


def read_litdata_ncv_for_mi(
        litdata_dir: str,
        output_dir: str,
        which_outer_test: int,
        which_inner_val: int,
        threshold_ptp: float=100.0,
        path_excutable: Optional[str] = None,
        thre_sd: float = 0.05,
        thre_pcc: float = 0.9,
        thre_mi: float = 0.2,
    ) -> None:
    r"""Read specific NCV litdata from directories and calculate MI for each inner training set.

    Args:
        litdata_dir: Directory containing the LitData for nested cross-validation.
    
    """
    # Check if path_excutable is None
    if path_excutable is None:
        path_excutable = os.path.join(os.path.dirname(__file__), "mi2graph")
    if not os.path.exists(path_excutable):
        raise FileNotFoundError(f"Executable file not found: {path_excutable}")

    # Read litdata
    dir_xoi = os.path.join(litdata_dir, f"ncv_test_{which_outer_test}_val_{which_inner_val}")
    dataloader_trn = StreamingDataLoader(StreamingDataset(os.path.join(dir_xoi, const.dkey.title_train)))

    # Run the compiled MI-based proccessing procedure on training data
    trnset_npy_dir = os.path.join(output_dir, "tmp_trnset")
    os.makedirs(trnset_npy_dir, exist_ok=True)
    mi_net_dir = os.path.join(output_dir, "mi_net_for_traindataset")
    os.makedirs(mi_net_dir, exist_ok=True)
    
    # Read (multiple) omics' data from dataloader_trn and save it to a matrix, then save the matrix to a NPY file.
    trn_data_matrix = [omics_tensor_list_to_np(batch['omics']) for batch in dataloader_trn]
    trn_data_matrix = np.array(trn_data_matrix).astype(np.float64).transpose()
    
    # Scale and log2 transformation for each feature
    trn_data_matrix, values_min, values_max, feat2rm = auto_proc_feat4trn(trn_data_matrix, threshold_ptp)

    # Check if the matrix contains 'None' value
    if np.any(np.isnan(trn_data_matrix)):
        raise ValueError("The matrix contains 'None' value.")
    
    # Save the matrix to a NPY file
    path_npy = os.path.join(trnset_npy_dir, f"trn_{which_outer_test}_{which_inner_val}.npy")
    np.save(path_npy, trn_data_matrix)
    
    # Save the min and max values for scaling back & scaling validation and testing dataset
    path_range = os.path.join(mi_net_dir, f"trn_{which_outer_test}_{which_inner_val}_range.npz")
    np.savez(path_range, values_min=values_min, values_max=values_max, feat2rm=feat2rm)
    
    # Run
    path_npz = os.path.join(mi_net_dir, f"trn_{which_outer_test}_{which_inner_val}.npz")
    cmd_mi = f"{path_excutable} -i {path_npy} -o {path_npz} --thresd {thre_sd} --threpcc {thre_pcc} --thremi {thre_mi}"
    print("\nRUNNING: ", cmd_mi, "\n")
    os.system(cmd_mi)

    os.remove(path_npy)
    return None


def train_model(
        model: Any,
        datamodule: LightningDataModule | Any,
        es_patience: int,
        max_epochs: int,
        min_epochs: int,
        log_dir: str,
        # devices: Union[list[int], str, int] = const.default.devices,
        accelerator: str = const.default.accelerator,
        in_dev: bool = False,
    ):
    r"""Fit the model.
    """
    # avail_dev = get_avail_nvgpu(devices)

    callback_es = EarlyStopping(
        monitor=const.dkey.title_val_loss,
        patience=es_patience,
        mode='min',
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename=const.default.ckpt_fname_format,
        monitor=const.dkey.title_val_loss,
    )

    logger_tr = TensorBoardLogger(save_dir=log_dir, name='')

    trainer = Trainer(
        fast_dev_run=in_dev,
        logger=logger_tr,
        log_every_n_steps=1,
        precision="16-mixed",
        # devices=avail_dev,
        accelerator=accelerator,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[callback_es, callback_ckpt],
        num_sanity_val_steps=0,
        default_root_dir=log_dir,
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
