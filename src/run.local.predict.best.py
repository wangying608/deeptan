import os
import sys

import polars as pl

import deeptan.constants as const
from deeptan.graph.recon import predict

# /home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python run.local.predict.best.py

# _seed = "seed_42"
_seed = f"seed_{sys.argv[1] if len(sys.argv) > 1 else 42}"
path_best_ckpt_seed_xx = f"/home/wuch/prjs/git_nwafu/DeepTAN/src/.collected_logs/best_ckpts_{_seed}.parquet"

home_deeptan = "/mnt/hdd2/homext/wuch/xn2p"
data_dir = os.path.join(home_deeptan, "data", "optimized_data")
output_dir = os.path.join(home_deeptan, "run", "predict", "deeptan")

splits = const.dkey.splits
# splits = [const.dkey.abbr_test]


if __name__ == "__main__":
    best_ckpt_seed_xx = pl.read_parquet(path_best_ckpt_seed_xx)
    for _split in splits:
        for _row in best_ckpt_seed_xx.iter_rows(named=True):
            _ckpt_path = _row["ckpt_path"]
            _litdata_dir = os.path.join(data_dir, _row["data"], _row["seed"], _split)
            _output_path = os.path.join(output_dir, _row["data"], f"preds+{_row['seed']}+{_row['task']}+{_split}.h5")
            if os.path.exists(_output_path):
                print(f"Output file already exists: {_output_path}")
                continue

            print("\nPredicting: ", _ckpt_path, _litdata_dir)
            try:
                predict(model_ckpt_path=_ckpt_path, litdata_dir=_litdata_dir, output_path=_output_path, map_location=None, batch_size=8, save_h5=True)
                print("Results saved to: ", _output_path, "\n")
            except Exception as e:
                print(f"Error occurred while predicting: {_ckpt_path}, {_litdata_dir}. Error: {e}")
                continue
