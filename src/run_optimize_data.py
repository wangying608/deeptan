import os
import argparse
import json
import pickle
import litdata
from deeptan.utils.uni import get_avail_cpu_count
from deeptan.utils.data import DeepTANDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTAN pipeline for training and testing.")
    
    parser.add_argument('--labels', type=str, default="", help='Path to label data in .parquet format')
    parser.add_argument('--bs', type=int, default=1, help='Batch size for training')
    parser.add_argument('--trn_npz', type=str, required=True, help='Path to training data in .npz format')
    parser.add_argument('--val_parquet', type=str, required=True, help='Path to validation data in .parquet format')
    parser.add_argument('--tst_parquet', type=str, required=True, help='Path to test data in .parquet format')
    parser.add_argument('--output_dir', type=str, default=".tmp_data_optimized", help='Directory for logging')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if len(args.labels) < 2:
        labels = None
    else:
        labels = args.labels
    
    files_fit = {"trn": args.trn_npz, "val": args.val_parquet, "tst": args.tst_parquet}
    datamodule = DeepTANDataModule(files_fit, labels, batch_size=args.bs)
    datamodule.setup()

    # Copy original training data to output directory for saving node_names and g_label_dim
    # shutil.copy(args.trn_npz, os.path.join(args.output_dir, "trn.npz"))
    others2save = {
        "dict_node_names": datamodule.dict_node_names,
        "output_g_label_dim": datamodule.label_dim,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    # Save as json
    with open(os.path.join(args.output_dir, "others2save.json"), "w") as f:
        json.dump(others2save, f)
    # Save as pickle
    with open(os.path.join(args.output_dir, "others2save.pkl"), "wb") as f:
        pickle.dump(others2save, f)

    # Optimize
    litdata.optimize(
        fn = datamodule.train.get,
        inputs = list(range(datamodule.train.len())),
        output_dir = os.path.join(args.output_dir, "trn"),
        chunk_bytes = "512MB",
        compression = "zstd",
        num_workers = get_avail_cpu_count(24),
    )
    litdata.optimize(
        fn = datamodule.val.get,
        inputs = list(range(datamodule.val.len())),
        output_dir = os.path.join(args.output_dir, "val"),
        chunk_bytes = "512MB",
        compression = "zstd",
        num_workers = get_avail_cpu_count(24),
    )
    litdata.optimize(
        fn = datamodule.test.get,
        inputs = list(range(datamodule.test.len())),
        output_dir = os.path.join(args.output_dir, "tst"),
        chunk_bytes = "512MB",
        compression = "zstd",
        num_workers = get_avail_cpu_count(24),
    )
