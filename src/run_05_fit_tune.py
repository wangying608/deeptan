r"""
DeepTAN hyperparameter tuning class with Optuna integration.
"""

import argparse

import deeptan.constants as const
from deeptan.graph.model import DeepTANTune


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTAN tuning pipeline.")

    parser.add_argument("--litdata", type=str, default="", required=False, help="Path to litdata directory")
    parser.add_argument("--bs", type=int, default=const.default.bs, help="Batch size for training")
    parser.add_argument("--log_dir", type=str, default=".tmp_logs_tune", help="Directory for logging")
    parser.add_argument("--input_node_emb_dim", type=int, default=1, help="Input node embedding dimension")
    parser.add_argument("--is_regression", action="store_true", help="Whether the task is regression")
    parser.add_argument("--acc_grad_batch", type=int, default=const.default.accumulate_grad_batches, help="Accumulate gradients over multiple batches")
    parser.add_argument("--chunk_size", type=int, default=const.default.chunk_size, help="A proper chunk size can balance memory usage and speed")
    parser.add_argument("--accelerator", type=str, default=const.default.accelerator, help="cpu, gpu, tpu, hpu, mps, auto")
    parser.add_argument("--ntrials", type=int, default=const.default.n_trials, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--njobs", type=int, default=const.default.n_jobs, help="The number of parallel jobs for Optuna. If this argument is set to -1, the number is set to CPU count.")

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_args()

    # Example configuration (should match original argparse parameters)
    config = {
        "log_dir": args.log_dir,
        "litdata": args.litdata,
        "bs": args.bs,
        "chunk_size": args.chunk_size,
        "is_regression": args.is_regression,
        "accelerator": args.accelerator,
        "input_node_emb_dim": args.input_node_emb_dim,
        "acc_grad_batch": args.acc_grad_batch,
        "es": const.default.es,
        "node_emb_dim": 128,
        "fusion_dims_node_emb": [64, 32, 16],
        "output_dim_g_emb": 192,
        "n_hop": 2,
        "threshold_edge_exist": 0.05,
        "threshold_subgraph_overlap": 0.99,
        "n_heads_node_emb": 2,
        "n_heads_pooling": 2,
        "dropout": const.default.dropout,
        "lr": const.default.lr,
        "max_ep": 1000,
        "min_ep": 2,
        "nworker": const.default.n_workers,
    }

    tuner = DeepTANTune(config)
    tuner.optimize(n_trials=args.ntrials, n_jobs=args.njobs)
