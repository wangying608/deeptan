r"""
Default values.
"""

from os import getenv
from numpy import ceil
from multiprocessing import cpu_count


lr = 0.0001
negative_slope = 0.2
node_feature_dim = 32
hidden_dim_dyn_cen = 64

matmul_precision = "medium"
accelerator = "auto"
devices = "auto"

n_threads = int(getenv("NUM_THREADS", ceil(cpu_count() * 0.8)))

time_format = "%Y%m%d%H%M%S"
time_delay = 11.7
ckpt_fname_format = "best-model-{epoch:04d}-{val_loss:.4f}"
optuna_db = "sqlite:///optuna.db"
n_jobs = 1
n_trials = 10
n_workers = 1
