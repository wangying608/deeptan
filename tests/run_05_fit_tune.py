r"""
DeepTAN hyperparameter tuning class with Optuna integration.
"""

import argparse

import deeptan.constants as const
from deeptan.graph.model import DeepTANTune


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTAN tuning pipeline.")
    parser.add_argument("--auto_tune", "--atune", action="store_true", help="Whether to perform hyperparameter tuning")

    parser.add_argument("--em", type=str, default="", help="Existing model checkpoint path for loading")

    parser.add_argument("--litdata", "--data", type=str, required=False, help="Path to litdata directory")
    parser.add_argument("--bs", type=int, default=const.default.bs, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=const.default.lr, help="Learning rate")
    parser.add_argument("--log_dir", "--logdir", type=str, default=".tmp_logs_tune", help="Directory for logging")
    parser.add_argument("--input_node_emb_dim", "--indim", type=int, default=1, help="Input node embedding dimension")
    parser.add_argument("--is_regression", "--ir", action="store_true", help="Whether the task is regression")
    parser.add_argument("--acc_grad_batch", "--agb", type=int, default=const.default.accumulate_grad_batches, help="Accumulate gradients over multiple batches")
    parser.add_argument("--chunk_size", "--ck", type=int, default=const.default.chunk_size, help="A proper chunk size can balance memory usage and speed")
    parser.add_argument("--accelerator", "--ac", type=str, default=const.default.accelerator, help="cpu, gpu, tpu, hpu, mps, auto")
    parser.add_argument("--ntrials", "--nt", type=int, default=const.default.n_trials, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--njobs", "--nj", type=int, default=const.default.n_jobs, help="The number of parallel jobs for Optuna. If this argument is set to -1, the number is set to CPU count.")

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_args()

    print(f"\nArguments: {args}\n")

    # Example configuration (should match original argparse parameters)
    config = {
        "log_dir": args.log_dir,
        "litdata": args.litdata,
        "bs": args.bs,
        "lr": args.lr,
        "chunk_size": args.chunk_size,
        "is_regression": args.is_regression,
        "accelerator": args.accelerator,
        "input_node_emb_dim": args.input_node_emb_dim,
        "acc_grad_batch": args.acc_grad_batch,
        "es": const.default.es,
        "node_emb_dim": const.default.node_emb_dim,
        "fusion_dims_node_emb": const.default.fusion_dims_node_emb,
        "output_dim_g_emb": const.default.g_emb_dim,
        "n_hop": const.default.n_hop,
        "threshold_edge_exist": const.default.threshold_edge_exist,
        "threshold_subgraph_overlap": const.default.threshold_subg_overlap,
        "n_heads_node_emb": const.default.n_heads_node_emb,
        "n_heads_pooling": const.default.n_heads_pooling,
        "n_heads_ge_decoder": const.default.n_heads_ge_decoder,
        "n_heads_label_pred": const.default.n_heads_label_pred,
        "dropout": const.default.dropout,
        "max_ep": const.default.max_epoch,
        "min_ep": const.default.min_epoch,
        "nworker": const.default.n_workers,
    }

    if len(args.em) < 3:
        ckpt = None
    else:
        ckpt = args.em

    if args.auto_tune:
        tuner = DeepTANTune(config, ckpt)
        tuner.optimize(n_trials=args.ntrials, n_jobs=args.njobs)
    else:
        trainer = DeepTANTune(config, ckpt)
        trainer._train_on_args()
