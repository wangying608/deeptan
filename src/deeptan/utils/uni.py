r"""
Universal functions.
"""

import random
import string
import time
from multiprocessing import cpu_count
from typing import Optional

from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
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
