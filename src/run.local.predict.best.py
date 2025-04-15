import os

import polars as pl

from deeptan.graph.recon import predict

path_best_ckpt_seed_xx = "/home/wuch/prjs/git_nwafu/DeepTAN/src/.collected_logs/best_ckpts_seed_42.parquet"

home_deeptan = "/mnt/hdd2/homext/wuch/xn2p"
data_dir = os.path.join(home_deeptan, "data", "optimized_data")
output_dir = os.path.join(home_deeptan, "run", "predict")
trnvaltst = "tst"


if __name__ == "__main__":
    best_ckpt_seed_xx = pl.read_parquet(path_best_ckpt_seed_xx)
    for _row in best_ckpt_seed_xx.iter_rows(named=True):
        _ckpt_path = _row["ckpt_path"]
        _litdata_dir = os.path.join(data_dir, _row["data"], _row["seed"], trnvaltst)
        # _output_pkl_path = os.path.join(output_dir, _row["data"], _row["seed"], _row["task"], "preds.pkl")
        _output_pkl_path = os.path.join(output_dir, _row["data"], f"preds+{_row['seed']}+{_row['task']}+{trnvaltst}+{_row['log_name']}.pkl")

        print("\nPredicting: ", _ckpt_path, _litdata_dir)
        try:
            predict(model_ckpt_path=_ckpt_path, litdata_dir=_litdata_dir, output_pkl_path=_output_pkl_path, map_location=None, batch_size=8)
            print("Results saved to: ", _output_pkl_path, "\n")
        except Exception as e:
            print(f"Error occurred while predicting: {_ckpt_path}, {_litdata_dir}. Error: {e}")
            continue
