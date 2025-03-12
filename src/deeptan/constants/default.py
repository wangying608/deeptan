r"""
Default values.
"""

from multiprocessing import cpu_count
from os import getenv

from numpy import ceil

bs = 32
accumulate_grad_batches = 1
lr = 0.0005
es = 2
min_epoch = 2
max_epoch = 10
chunk_size = 1024
subg_chunk_size = 4
dropout = 0.0
negative_slope = 0.2
node_emb_dim = 128
g_emb_dim = 256
label_pred_hidden_dims = [512, 256, 256]
fusion_dims_node_emb = [128, 64]
n_heads_pooling = 4
n_heads_node_emb = 2
n_heads_ge_decoder = 2
n_heads_label_pred = 2
n_hop = 1

threshold_subg_overlap = 0.95
threshold_edge_exist = 0.3

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

lit_chunk_bytes = "256MB"
lit_compression = "zstd"
