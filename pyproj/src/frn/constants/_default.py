"""
Default values.
"""

time_format = "%Y%m%d%H%M%S"
time_delay = 11.7
ckpt_fname_format = "best-model-{epoch:04d}-{val_loss:.4f}"
optuna_db = "sqlite:///optuna.db"
n_jobs = 1
n_trials = 10
n_workers = 0
n_workers_litdata = 1
accelerator = "auto"
devices = "auto"
float32_matmul_precision = "high"
compression_alg = "zstd"
chunk_bytes = "256MB"

seed_1 = 42
seed_2 = 43

lr = 1e-4
batch_size = 32
max_epochs = 1000
min_epochs = 20
patience = 20
dropout = 0.4

hidden_dim = 1024
n_encoders = 2
n_heads = 2

snp_onehot_bits = 10

# Graph
negative_slope = 0.2
node_feature_dim = 32
