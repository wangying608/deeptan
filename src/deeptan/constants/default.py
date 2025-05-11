r"""
Default values.
"""

from multiprocessing import cpu_count
from os import getenv

from numpy import ceil

bs = 16
accumulate_grad_batches = 2
lr = 0.0002
es = 5
min_epoch = 3
max_epoch = 50
dropout = 0.3
negative_slope = 0.2
node_emb_dim = 128
g_emb_dim = 256
label_pred_hidden_dims = [512, 256, 256]
fusion_dims_node_emb = [128, 64]
n_heads_pooling = 8
n_heads_node_emb = 8
n_heads_ge_decoder = 8
n_heads_label_pred = 8
n_hop = 1

chunk_size = 2048
mem_safety_factor = 0.85
operation_overhead = 3.3

threshold_nmic = 0.3
threshold_subg_overlap = 0.95
threshold_edge_exist = 0.2

matmul_precision = "high"
accelerator = "auto"
devices = "auto"
precision = "16-mixed"

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

model_config = {
    "guide_gat": True,
    "class_weights": None,
    "use_focal_loss": True,
    "focal_alpha": None,
    "node_emb_dim": node_emb_dim,
    "fusion_dims_node_emb": fusion_dims_node_emb,
    "output_dim_g_emb": g_emb_dim,
    "n_hop": n_hop,
    "threshold_edge_exist": threshold_edge_exist,
    "threshold_subgraph_overlap": threshold_subg_overlap,
    "n_heads_node_emb": n_heads_node_emb,
    "n_heads_pooling": n_heads_pooling,
    "n_heads_ge_decoder": n_heads_ge_decoder,
    "n_heads_label_pred": n_heads_label_pred,
    "dropout": dropout,
    "lr": lr,
    "chunk_size": chunk_size,
    "n_workers": n_workers,
}
