import os
import sys

import polars as pl

import deeptan.constants as const
from deeptan.graph.recon import predict

# /home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python run.local.predict.best.py

_seed = f"seed_{sys.argv[1] if len(sys.argv) > 1 else 42}"
# path_best_ckpt_seed_xx = f"/home/wuch/prjs/git_nwafu/DeepTAN/src/.collected_logs/best_ckpts_{_seed}.parquet"
path_best_ckpt_seed_xx = f"/home/wuch/prjs/git_nwafu/DeepTAN/src/.collected_logs/best_ckpts_{_seed}.csv"

home_deeptan = "/mnt/hdd2/homext/wuch/xn2p"
data_dir = os.path.join(home_deeptan, "data", "optimized_data")
output_dir = os.path.join(home_deeptan, "run", "predict", "deeptan")

_split = const.dkey.abbr_test

# ns_feat = [400, 800, 1200, 1600, 2000]
ns_feat = [400, 800, 1200, 1600]


if __name__ == "__main__":
    # best_ckpt_seed_xx = pl.read_parquet(path_best_ckpt_seed_xx)
    best_ckpt_seed_xx = pl.read_csv(path_best_ckpt_seed_xx)
    for _row in best_ckpt_seed_xx.iter_rows(named=True):
        if _row["task"] != "multitask":
            continue
        _ckpt_path = _row["ckpt_path"]

        for _n_feat in ns_feat:
            _dataset = _row["data"].split("top")[0] + f"top{_n_feat}"
            _litdata_dir = os.path.join(data_dir, _dataset, _row["seed"], _split)
            _output_pkl_path = os.path.join(output_dir, _row["data"] + f"+top_{_n_feat}", f"preds+{_row['seed']}+{_row['task']}+{_split}.pkl")
            if os.path.exists(_output_pkl_path):
                print(f"Output file already exists: {_output_pkl_path}")
                continue

            print("\nPredicting: ", _ckpt_path, _litdata_dir)
            try:
                predict(model_ckpt_path=_ckpt_path, litdata_dir=_litdata_dir, output_pkl_path=_output_pkl_path, map_location=None, batch_size=8)
                print("Results saved to: ", _output_pkl_path, "\n")
            except Exception as e:
                print(f"Error occurred while predicting: {_ckpt_path}, {_litdata_dir}. Error: {e}")
                continue
