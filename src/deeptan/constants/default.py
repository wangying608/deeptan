r"""
Default values.
"""

from os import getenv
from numpy import ceil
from multiprocessing import cpu_count


bs = 32
accumulate_grad_batches = 1
lr = 0.0001
es = 5
chunk_size = 8192
dropout = 0.1
negative_slope = 0.2

matmul_precision = "high"
accelerator = "auto"
devices = "auto"

n_threads = int(getenv("NUM_THREADS", ceil(cpu_count() * 0.8)))

time_format = "%Y%m%d%H%M%S"
time_delay = 11.7
ckpt_fname_format = "best-model-{epoch:04d}-{val_loss:.4f}"
optuna_db = "sqlite:///optuna.db"
n_jobs = 1
n_trials = 30
n_workers = 1

lit_chunk_bytes = "512MB"
lit_compression = "zstd"
