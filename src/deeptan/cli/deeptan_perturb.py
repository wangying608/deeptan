import argparse
import os

from deeptan.graph.recon import predict_perturbation


def deeptan_perturb():
    parser = argparse.ArgumentParser(description="DeepTAN perturbation script.")
    parser.add_argument("--em", type=str, required=True, help="Existing model checkpoint path.")
    parser.add_argument("--litdata", "--data", type=str, required=True, help="Path to litdata directory")
    parser.add_argument("--output", "--out", type=str, required=True, help="Path to output file")
    parser.add_argument("--maplocation", "--maploc", type=str, default=None, help="Map location for model loading")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    model_path = args.em
    litdata_dir = args.litdata

    # Create output directory
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Predicting with model {model_path} on data {litdata_dir}")
    predict_perturbation(
        model_ckpt_path=model_path,
        litdata_dir=litdata_dir,
        output_path=args.output,
        n_perturbations=5,
        map_location=args.maplocation,
        overwrite_files=args.overwrite,
    )
