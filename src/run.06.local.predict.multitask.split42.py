import os
import sys

import deeptan.constants as const
from deeptan.graph.recon import predict

_seed = "seed_42"
# _seed = f"seed_{sys.argv[1] if len(sys.argv) > 1 else 42}"
home_deeptan = "/mnt/hdd2/homext/xn2p"
dataset = "ath_pretrain.full.nmicg8"
task_name = "multitask"
ckpt_name = os.path.join("DeepTAN_20260306094539_zjCFC", "best_model.ckpt")

batch_size = 8
splits = [const.dkey.abbr_test]

logs_dir = os.path.join(home_deeptan, "run", "logs_wy")  # Model checkpoint directory
output_dir = os.path.join(home_deeptan, "run", "predict", "deeptan")
data_dir = os.path.join(home_deeptan, "data", "optimized_new")
ckpt_path = os.path.join(logs_dir, dataset, _seed, task_name, ckpt_name)
dataset_dir = os.path.join(data_dir, dataset, _seed)


if __name__ == "__main__":
    for _split in splits:
        _ckpt_path = ckpt_path
        _litdata_dir = os.path.join(dataset_dir, _split)
        _output_path = os.path.join(output_dir, dataset, f"preds+{_seed}+{task_name}+{_split}.h5")
        if os.path.exists(_output_path):
            print(f"Output file already exists: {_output_path}")
            continue

        print("\nPredicting: ", _ckpt_path, _litdata_dir)
        predict(
            model_ckpt_path=_ckpt_path,
            litdata_dir=_litdata_dir,
            output_path=_output_path,
            map_location=None,
            batch_size=batch_size,
        )
