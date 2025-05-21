r"""
Universal functions.
"""

import concurrent.futures
import os
import pickle
import random
import string
import time
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import optuna
import polars as pl
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import cuda
from torch_geometric.data import Batch

import deeptan.constants as const


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


class GetAdaptiveChunkSize:
    def __init__(
        self,
        mem_safety_factor: Optional[float] = None,
        operation_overhead: Optional[float] = None,
        min_chunk_size: int = 2,
        total_vram: Optional[int] = None,
    ):
        self.mem_safety_factor = mem_safety_factor if mem_safety_factor is not None else const.default.mem_safety_factor
        self.operation_overhead = operation_overhead if operation_overhead is not None else const.default.operation_overhead
        self.min_chunk_size = min_chunk_size

        self.total_mem = 0
        if cuda.is_available():
            self.total_mem = sum(cuda.mem_get_info(device=i)[1] for i in range(cuda.device_count()))

        if total_vram is not None:
            self.total_mem = total_vram * 1024 * 1024 * 1024
        else:
            # Try to read total VRAM from environment variable
            _total_mem = os.getenv("TOTAL_VRAM", None)
            if _total_mem is not None:
                try:
                    self.total_mem = int(_total_mem) * 1024 * 1024 * 1024
                except ValueError:
                    pass

        # print(f"Total VRAM: {self.total_mem / (1024 * 1024 * 1024):.2f} GB")

    def calc(self, tensor_shape: Tuple[int, ...], dim: int = 0, use_total_as_avail: bool = True) -> int:
        required_mem = self.estimate_tensor_memory(tensor_shape)
        if required_mem == 0 or self.total_mem == 0:
            return const.default.chunk_size

        if use_total_as_avail:
            max_allowed_mem = self.total_mem * self.mem_safety_factor / self.operation_overhead
        else:
            max_allowed_mem = sum(cuda.mem_get_info(device=i)[0] for i in range(cuda.device_count())) * self.mem_safety_factor / self.operation_overhead

        if required_mem > max_allowed_mem:
            n_chunks = np.ceil(required_mem / max_allowed_mem)
        else:
            n_chunks = 1
        chunk_size = max(self.min_chunk_size, int(tensor_shape[dim] // n_chunks))
        # print(f"Chunk size {chunk_size} for tensor shape {tensor_shape}, Required memory: {required_mem / (1024**3)} GB, Max allowed memory: {max_allowed_mem / (1024**3)} GB, Total memory: {self.total_mem / (1024**3)} GB.")
        return chunk_size

    def estimate_tensor_memory(self, tensor_shape: Tuple[int, ...], dtype_size: int = 4) -> int:
        """Estimate memory (bytes) required for a tensor given its shape."""
        return int(np.prod(tensor_shape) * dtype_size)


def get_map_location(map_loc: Optional[str] = None):
    if map_loc is None:
        if cuda.device_count() > 0:
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


def process_ckpt_path(path_x: str) -> pl.DataFrame | None:
    """Process a single checkpoint path and return the corresponding DataFrame."""
    tsb_dir = os.path.join(os.path.dirname(path_x), "version_0")
    tsb_event = read_tensorboard_events(tsb_dir, False)
    assert isinstance(tsb_event, Dict), "tsb_event must be a dictionary."
    _df = tsbevent2df(tsb_event)
    if _df.width > 0:
        path_x_frag = path_x.split(os.sep)
        if path_x_frag[-2].startswith("trial_"):
            posmv = 1
        else:
            posmv = 0
        _log_name = path_x_frag[-2 - posmv]
        _task = path_x_frag[-3 - posmv]
        _seed = path_x_frag[-4 - posmv]
        _data = path_x_frag[-5 - posmv]

        _info_df = pl.DataFrame({"ckpt_path": [path_x], "log_name": [_log_name], "task": [_task], "seed": [_seed], "data": [_data]})
        return _info_df.hstack(_df)
    else:
        print(f"No records found in {tsb_dir}. Skipping...")
        return None


def collect_tensorboard_events(dir_log: str) -> pl.DataFrame:
    r"""Collect info from tensorboard events."""
    paths_ckpt = search_ckpt(dir_log)
    records: List[pl.DataFrame] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=const.default.n_threads) as executor:
        futures = [executor.submit(process_ckpt_path, path_x) for path_x in paths_ckpt]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                records.append(result)

    if len(records) == 0:
        raise ValueError("No records found.")
    return pl.concat(records, how="diagonal", rechunk=True)


def search_ckpt(dir_log: str):
    r"""Search checkpoints in the directory and its subdirectories."""
    paths_ckpt = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(dir_log) for f in files if f.endswith(".ckpt")]
    if len(paths_ckpt) == 0:
        raise FileNotFoundError("No checkpoint files found.")
    paths_ckpt.sort()
    print(f"Found {len(paths_ckpt)} checkpoints.\n")
    return paths_ckpt


def read_tensorboard_events(dir_events: str, get_test_loss: bool = True) -> Dict[str, Any] | float:
    r"""Read tensorboard events from the directory."""
    event_acc = EventAccumulator(dir_events)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()["scalars"]
    scalar_data = {tag: [] for tag in scalar_tags}

    for tag in scalar_tags:
        _events = event_acc.Scalars(tag)
        for _event in _events:
            scalar_data[tag].append((_event.step, _event.value))

    if get_test_loss:
        test_loss: float = scalar_data[const.dkey.title_tst_loss][0][1]
        return test_loss
    else:
        return scalar_data


def tsbevent2df(tsbevent: Dict, keys: Optional[List[str]] = None):
    r"""
    Convert tensorboard event to polars dataframe.
    """
    if keys is None:
        keys = const.dkey.tsb_keys2pick
    _tsb_metrics = {_key: tsbevent[_key][0][1] for _key in keys if _key in tsbevent.keys()}
    dtype_dict = {col: pl.Float64 for col in _tsb_metrics.keys()}
    _tsb_metrics_df = pl.DataFrame(_tsb_metrics, schema=dtype_dict)
    # colnames_new = ["_".join(_n.split("_")[1:]) for _n in keys]
    # colnames_new = [_n.split("/")[1] for _n in _tsb_metrics.keys()]
    # _tsb_metrics_df.columns = colnames_new
    return _tsb_metrics_df


def collect_optuna_db(dir_log: str):
    r"""Collect info of optuna db files.
    This function will find all optuna db files in the directory `dir_log` and its subdirectories,
    and read the info of each optuna db file into a dataframe.
    """
    paths_optuna_db = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(dir_log) for f in files if f.endswith(".db")]
    if len(paths_optuna_db) == 0:
        raise FileNotFoundError("No optuna db files found.")
    paths_optuna_db.sort()
    print(f"Found {len(paths_optuna_db)} optuna db files\n")

    # Read optuna db files and store the results in a dataframe
    studies_dicts = [read_optuna_db(path_optuna_db) for path_optuna_db in paths_optuna_db]
    studies_df = pl.DataFrame(studies_dicts)

    return studies_df


def read_optuna_db(path_optuna_db: str) -> Dict[str, Any]:
    loaded_study = optuna.load_study(study_name=None, storage=f"sqlite:///{path_optuna_db}")
    study_name = loaded_study.study_name
    min_loss = loaded_study.best_value
    trials_df = loaded_study.trials_dataframe()
    best_trial = loaded_study.best_trial
    best_params = best_trial.params
    best_trial_duration = best_trial.duration.total_seconds() if best_trial.duration is not None else None
    best_trial_datetime_start = best_trial.datetime_start.isoformat() if best_trial.datetime_start is not None else None
    return {
        "study_name": study_name,
        "min_loss": min_loss,
        "best_params": best_params,
        "best_trial_duration": best_trial_duration,
        "best_trial_datetime_start": best_trial_datetime_start,
        "trials_df": trials_df,
    }


def save_to_h5(
    data_dict: dict,
    file_path: str,
    mode: str = "a",
    compression: bool = True,
    group_path: Optional[str] = None,
) -> None:
    """
    Save a dictionary to an HDF5 file, handling various data types including:
    - numpy arrays (with optional compression)
    - string arrays/lists
    - nested dictionaries
    - other numeric data

    Args:
        data_dict: Dictionary containing data to save
        file_path: Output HDF5 file path
        mode: Mode in which to open the file ('a' for append, 'w' for overwrite)
        compression: Whether to use compression for arrays
        group_path: Path to the group where data should be saved (e.g., '/group/subgroup')
    """
    with h5py.File(file_path, mode) as f:
        target = f
        if group_path:
            target = target.require_group(group_path)
        for key, value in data_dict.items():
            try:
                if isinstance(value, np.ndarray):
                    if value.dtype.kind == "U":
                        _dt = h5py.string_dtype(encoding="utf-8")
                        target.create_dataset(key, data=value, dtype=_dt, compression="gzip" if compression else None)
                    else:
                        target.create_dataset(key, data=value, compression="gzip" if compression else None)
                elif isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
                    _dt = h5py.string_dtype(encoding="utf-8")
                    target.create_dataset(key, data=value, dtype=_dt)
                elif isinstance(value, dict):
                    sub_grp = target.create_group(key)
                    if value:
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (list, tuple)) and all(isinstance(x, str) for x in subvalue):
                                _dt = h5py.string_dtype(encoding="utf-8")
                                sub_grp.create_dataset(subkey, data=subvalue, dtype=_dt)
                            else:
                                sub_grp.create_dataset(subkey, data=subvalue)
                else:
                    target.create_dataset(key, data=value)
            except Exception as e:
                print(f"Warning: Failed to save {key} to HDF5: {str(e)}")
                continue


def path_exists_in_hdf5(file_path: str, hdf5_path: str) -> bool:
    """
    Check if a path exists in an HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        hdf5_path: Path within the HDF5 file (e.g., '/group/subgroup/dataset')

    Returns:
        bool: True if path exists, False otherwise
    """
    try:
        with h5py.File(file_path, "r") as f:
            return hdf5_path in f
    except (OSError, KeyError):
        return False


def convert_pickle_to_h5(input_pkl: str, output_h5: str):
    """
    Convert a pickle file to h5 format using the save_to_h5 utility.

    Args:
        input_pkl: Path to input pickle file
        output_h5: Path for output h5 file
    """
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"Input pickle file not found: {input_pkl}")
    # Load pickle data
    with open(input_pkl, "rb") as f:
        _data = pickle.load(f)

    # Save to h5 format
    _output_h5 = output_h5 if output_h5.endswith(".h5") else f"{output_h5}.h5"
    save_to_h5(_data, _output_h5, mode="w", compression=True)
