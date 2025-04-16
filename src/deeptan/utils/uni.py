r"""
Universal functions.
"""

import os
import random
import string
import time
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional

import optuna
import polars as pl
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.cuda import device_count
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


def print_nested_keys(d):
    for key in d:
        print(key)
        value = d[key]
        if isinstance(value, dict):
            print_nested_keys(value)


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


def collect_tensorboard_events(dir_log: str):
    r"""Collect info from tensorboard events."""
    paths_ckpt = search_ckpt(dir_log)

    # Pick ids of outer and inner folds, val_loss and version from ckpt file paths
    # records = [tsbevent2df(read_tensorboard_events(os.path.join(os.path.dirname(path_x), "version_0"), False)) for path_x in paths_ckpt]
    records = []
    for path_x in paths_ckpt:
        tsb_dir = os.path.join(os.path.dirname(path_x), "version_0")
        tsb_event = read_tensorboard_events(tsb_dir, False)
        assert isinstance(tsb_event, Dict), "tsb_event must be a dictionary."
        _df = tsbevent2df(tsb_event)
        if _df.width > 0:
            path_x_frag = path_x.split(os.sep)
            if path_x_frag[-2].startswith("trial_"):
                posmv = 1
                # _log_name = f"{path_x_frag[-3]}_{path_x_frag[-2]}"
            else:
                posmv = 0
                # _log_name = path_x_frag[-2]
            _log_name = path_x_frag[-2 - posmv]
            _task = path_x_frag[-3 - posmv]
            _seed = path_x_frag[-4 - posmv]
            _data = path_x_frag[-5 - posmv]

            _info_df = pl.DataFrame({"ckpt_path": [path_x], "log_name": [_log_name], "task": [_task], "seed": [_seed], "data": [_data]})
            # print(_info_df)
            _df = _info_df.hstack(_df)  # Concatenate the info DataFrame with the test records DataFrame
            records.append(_df)
        else:
            print(f"No records found in {tsb_dir}. Skipping...")

    # Convert to DataFrame by Polars
    if len(records) == 0:
        raise ValueError("No records found.")
    df = pl.concat(records, how="diagonal", rechunk=True)

    return df


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
    best_trial_duration = best_trial.duration.total_seconds()
    best_trial_datetime_start = best_trial.datetime_start.isoformat()
    return {
        "study_name": study_name,
        "min_loss": min_loss,
        "best_params": best_params,
        "best_trial_duration": best_trial_duration,
        "best_trial_datetime_start": best_trial_datetime_start,
        "trials_df": trials_df,
    }
