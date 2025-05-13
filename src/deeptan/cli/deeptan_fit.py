import argparse

import deeptan.constants as const
from deeptan.graph.model import DeepTANTune


def deeptan_fit_tune():
    parser = argparse.ArgumentParser(description="DeepTAN fitting and tuning pipeline.")
    parser.add_argument("--auto_tune", "--atune", action="store_true", help="Whether to perform hyperparameter tuning")

    parser.add_argument("--em", type=str, default="", help="Existing model checkpoint path for loading")
    parser.add_argument("--focus", type=str, default="None", help="Focus on a specific task, choose from 'None', 'recon', 'label', 'recon_and_freeze', 'label_and_freeze'")
    parser.add_argument("--no_guide_gat", "--nog", action="store_true", help="Whether to disable edge weights of guidance graphs on graph attentions")

    parser.add_argument("--litdata", "--data", type=str, required=False, help="Path to litdata directory")
    parser.add_argument("--bs", type=int, default=const.default.bs, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=const.default.lr, help="Learning rate")
    parser.add_argument("--log_dir", "--logdir", type=str, default=".tmp_logs_tune", help="Directory for logging")
    parser.add_argument("--input_node_emb_dim", "--indim", type=int, default=1, help="Input node embedding dimension")
    parser.add_argument("--is_regression", "--ir", action="store_true", help="Whether the task is regression")
    parser.add_argument("--acc_grad_batch", "--agb", type=int, default=const.default.accumulate_grad_batches, help="Accumulate gradients over multiple batches")
    parser.add_argument("--chunk_size", "--ck", type=int, default=const.default.chunk_size, help="A proper chunk size can balance memory usage and speed")
    parser.add_argument("--accelerator", "--ac", type=str, default=const.default.accelerator, help="cpu, gpu, tpu, hpu, mps, auto")
    parser.add_argument("--devices", "--dev", type=str, default=const.default.devices, help="Devices to use")
    parser.add_argument("--ntrials", "--nt", type=int, default=const.default.n_trials, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--njobs", "--nj", type=int, default=const.default.n_jobs, help="The number of parallel jobs for Optuna. If this argument is set to -1, the number is set to CPU count.")
    args = parser.parse_args()

    if args.no_guide_gat:
        guide_gat = False
    else:
        guide_gat = True

    _config = const.default.model_config.copy()
    _config.update(
        {
            "es": const.default.es,
            "max_ep": const.default.max_epoch,
            "min_ep": const.default.min_epoch,
            "log_dir": args.log_dir,
            "litdata": args.litdata,
            "bs": args.bs,
            "lr": args.lr,
            "chunk_size": args.chunk_size,
            "is_regression": args.is_regression,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "input_node_emb_dim": args.input_node_emb_dim,
            "acc_grad_batch": args.acc_grad_batch,
            "guide_gat": guide_gat,
        }
    )
    print(f"\n🔧Configuration⚙️\n{_config}\n")

    if len(args.em) < 3:
        ckpt = None
    else:
        ckpt = args.em

    if args.focus == "None":
        focus = None
    else:
        focus = args.focus

    trainer = DeepTANTune(_config, ckpt, focus)
    if args.auto_tune:
        trainer.optimize(n_trials=args.ntrials, n_jobs=args.njobs)
    else:
        trainer._train_on_args()
