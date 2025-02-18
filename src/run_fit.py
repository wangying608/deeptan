r"""
DeepTAN pipelines for fitting, hyperparameter tuning, inference, and testing.
"""

import os
import argparse
import pickle
import polars as pl
from deeptan.utils.uni import time_string, random_string
from deeptan.utils.data import DeepTANDataModule, DeepTANDataModuleLit, celltypes_class_weights
from deeptan.graph.model import DeepTAN, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTAN pipeline for training and testing.")
    
    parser.add_argument('--input_node_emb_dim', type=int, default=1, help='Input node embedding dimension')
    parser.add_argument('--labels', type=str, default="", help='Path to label data in .parquet format')
    parser.add_argument('--is_regression', action='store_true', help='Whether the task is regression')
    parser.add_argument('--onehot_class', type=str, default="", help='Path to a parquet file containing one-hot encoded class labels')
    parser.add_argument('--bs', type=int, default=4, help='Batch size for training')
    parser.add_argument('--acc_grad_batch', type=int, default=16, help='Accumulate gradients over multiple batches')
    parser.add_argument('--es_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--litdata', type=str, default="", required=False, help='Path to litdata directory')
    parser.add_argument('--trn_npz', type=str, default="", required=False, help='Path to training data in .npz format')
    parser.add_argument('--val_parquet', type=str, default="", required=False, help='Path to validation data in .parquet format')
    parser.add_argument('--tst_parquet', type=str, default="", required=False, help='Path to test data in .parquet format')
    parser.add_argument('--node_emb_dim', type=int, default=256, help='Node embedding dimension')
    parser.add_argument('--fusion_dims_node_emb', nargs='+', type=int, default=[256, 512, 256], help='Fusion dimensions for node embedding')
    parser.add_argument('--output_dim_g_emb', type=int, default=512, help='Output dimension for graph embedding')
    parser.add_argument('--n_hop', type=int, default=2, help='Number of hops')
    parser.add_argument('--threshold_edge_exist', type=float, default=0.1, help='Threshold for edge existence')
    parser.add_argument('--threshold_subgraph_overlap', type=float, default=0.99, help='Threshold for subgraph overlap')
    parser.add_argument('--heads_node_emb', type=int, default=2, help='Number of heads for node embedding')
    parser.add_argument('--heads_pooling', type=int, default=2, help='Number of heads for pooling')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='Negative slope for LeakyReLU')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for balancing loss terms')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--min_epochs', type=int, default=2, help='Minimum number of epochs')
    parser.add_argument('--log_dir', type=str, default=".tmp_logs", help='Directory for logging')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.log_dir.endswith("/"):
        args.log_dir = args.log_dir[:-1]
    args.log_dir = args.log_dir + "_" + time_string() + "_" + random_string(5)

    if len(args.litdata) > 0:
        # Read "others2save.pkl" from args.litdata
        with open(os.path.join(args.litdata, "others2save.pkl"), "rb") as f:
            others2save = pickle.load(f)
        dict_node_names = others2save["dict_node_names"]
        output_g_label_dim = others2save["output_g_label_dim"]
        
        datamodule = DeepTANDataModuleLit(args.litdata, batch_size=args.bs)
        datamodule.setup()
    elif len(args.trn_npz) > 0 and len(args.val_parquet) > 0 and len(args.tst_parquet) > 0:
        if len(args.labels) < 2:
            labels = None
        else:
            labels = args.labels
        files_fit = {"trn": args.trn_npz, "val": args.val_parquet, "tst": args.tst_parquet}
        datamodule = DeepTANDataModule(files_fit, labels, batch_size=args.bs)
        datamodule.setup()
        dict_node_names = datamodule.dict_node_names
        output_g_label_dim = datamodule.label_dim
    else:
        raise ValueError("Invalid arguments provided. Please check the input data paths and labels.")
    
    if len(args.onehot_class) < 2:
        class_weight = None
    else:
        class_weight = celltypes_class_weights(pl.read_parquet(args.onehot_class))

    # Initialize the model
    model = DeepTAN(
        dict_node_names=dict_node_names,
        input_dim=args.input_node_emb_dim,
        output_g_label_dim=output_g_label_dim,
        is_regression=args.is_regression,
        class_weights=class_weight,
        node_emb_dim=args.node_emb_dim,
        fusion_dims_node_emb=args.fusion_dims_node_emb,
        output_dim_g_emb=args.output_dim_g_emb,
        n_hop=args.n_hop,
        threshold_edge_exist=args.threshold_edge_exist,
        threshold_subgraph_overlap=args.threshold_subgraph_overlap,
        n_heads_node_emb=args.heads_node_emb,
        n_heads_pooling=args.heads_pooling,
        dropout=args.dropout,
        lr=args.lr,
        negative_slope=args.negative_slope,
        alpha=args.alpha,
    )

    train_model(
        model=model,
        datamodule=datamodule,
        es_patience=args.es_patience,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        log_dir=args.log_dir,
        accumulate_grad_batches=args.acc_grad_batch,
    )
