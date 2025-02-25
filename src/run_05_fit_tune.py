r"""
DeepTAN hyperparameter tuning class with Optuna integration.
"""

import os
import argparse
from deeptan.graph.model import DeepTANTune


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTAN tuning pipeline.")
    
    parser.add_argument('--litdata', type=str, default="", required=False, help='Path to litdata directory')
    parser.add_argument('--bs', type=int, default=8, help='Batch size for training')
    parser.add_argument('--log_dir', type=str, default=".tmp_logs_tune", help='Directory for logging')
    parser.add_argument('--input_node_emb_dim', type=int, default=1, help='Input node embedding dimension')
    parser.add_argument('--is_regression', action='store_true', help='Whether the task is regression')
    parser.add_argument('--onehot_class', type=str, default="", help='Path to a parquet file containing one-hot encoded class labels')
    parser.add_argument('--acc_grad_batch', type=int, default=8, help='Accumulate gradients over multiple batches')
    parser.add_argument('--chunk_size', type=int, default=1024, help='A proper chunk size can balance memory usage and speed')
    parser.add_argument('--accelerator', type=str, default="auto", help="cpu, gpu, tpu, hpu, mps, auto")

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
        "onehot_class": args.onehot_class,
        "accelerator": args.accelerator,
        "input_node_emb_dim": args.input_node_emb_dim,
        "acc_grad_batch": args.acc_grad_batch,
        "es": 5,
        "node_emb_dim": 128,
        "fusion_dims_node_emb": [256, 256, 256],
        "output_dim_g_emb": 256,
        "n_hop": 2,
        "threshold_edge_exist": 0.1,
        "threshold_subgraph_overlap": 0.99,
        "n_heads_node_emb": 2,
        "n_heads_pooling": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "negative_slope": 0.2,
        "alpha": 0.5,
        "max_ep": 1000,
        "min_ep": 2,
        "nworker": 1,
    }

    tuner = DeepTANTune(config)
    tuner.optimize(n_trials=50, n_jobs=1)
