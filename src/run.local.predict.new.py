import os
import sys

import polars as pl

import deeptan.constants as const
from deeptan.graph.recon_new import predict

# /home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python run.local.predict.best.py

# _seed = "seed_42"
_seed = f"seed_{sys.argv[1] if len(sys.argv) > 1 else 42}"

home_deeptan = "/mnt/hdd2/homext/wuch/xn2p"
data_dir = os.path.join(home_deeptan, "data", "optimized_data")
output_dir = os.path.join(home_deeptan, "run", "predict", "deeptan")

# splits = const.dkey.splits
splits = [const.dkey.abbr_test]


if __name__ == "__main__":
    for _split in splits:
        _ckpt_path = "/mnt/hdd2/homext/wuch/xn2p/run/logs/sc_multiome_minmi0.35_top2000/seed_42/multitask_new/DeepTAN_20250519162057_Vg3SP/best-model-epoch=0020-val_loss=0.0000.ckpt"
        _litdata_dir = os.path.join(data_dir, "sc_multiome_minmi0.35_top2000", _seed, _split)
        _output_path = os.path.join(output_dir, "sc_multiome_minmi0.35_top2000", f"preds+{_seed}+{'multitask_new'}+{_split}.h5")
        if os.path.exists(_output_path):
            print(f"Output file already exists: {_output_path}")
            continue

        print("\nPredicting: ", _ckpt_path, _litdata_dir)
        predict(
            model_ckpt_path=_ckpt_path,
            litdata_dir=_litdata_dir,
            output_path=_output_path,
            map_location=None,
            batch_size=32,
            save_h5=False,
        )
