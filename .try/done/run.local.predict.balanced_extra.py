import os
import sys

import deeptan.constants as const
from deeptan.graph.recon_diffpool import predict

# _seed = "seed_42"
_seed = f"seed_{sys.argv[1] if len(sys.argv) > 1 else 42}"

home_deeptan = "/mnt/hdd2/homext/wuch/xn2p"
output_dir = os.path.join(home_deeptan, "run", "predict", "deeptan")

data_dir = os.path.join(home_deeptan, "data", "optimized_data")
dataset = "snrna_full_extra"
dataset_dir = os.path.join(data_dir, dataset, _seed)
task_name = "multitask_diffpool"

logs_dir = "/mnt/hdd2/homext/wuch/xn2p/run/logs"

splits = const.dkey.splits
# splits = [const.dkey.abbr_test]
# splits = [const.dkey.abbr_train, const.dkey.abbr_val]


if __name__ == "__main__":
    for _split in splits:
        _ckpt_path = os.path.join(logs_dir, "snrna_full", _seed, task_name, "DeepTAN_20250526011113_xU5rq/best-model-epoch=0035-val_loss=0.0000.ckpt")
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
            batch_size=32,
        )
