import os
import sys

import deeptan.constants as const
from deeptan.graph.recon import predict

# _seed = "seed_42"
_seed = f"seed_{sys.argv[1] if len(sys.argv) > 1 else 42}"
home_deeptan = "/mnt/hdd2/homext/wuch/xn2p"
dataset = "snrna_6tissues_Epidermal.balanced.full.nmicg1"
task_name = "multitask"
ckpt_name = os.path.join("DeepTAN_20250610085339_RElWs", "trial_5", "best-model-epoch=0008.ckpt")
batch_size = 8
# splits = const.dkey.splits
splits = [const.dkey.abbr_test]

logs_dir = os.path.join(home_deeptan, "run", "logs")  # Model checkpoint directory
output_dir = os.path.join(home_deeptan, "run", "predict", "deeptan")
data_dir = os.path.join(home_deeptan, "data", "optimized_data")
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
