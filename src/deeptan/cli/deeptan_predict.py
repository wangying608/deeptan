import argparse
import os

from deeptan.graph.recon import predict


def deeptan_predict():
    parser = argparse.ArgumentParser(description="DeepTAN prediction script.")
    parser.add_argument("--em", type=str, required=True, help="Existing model checkpoint path.")
    parser.add_argument("--litdata", "--data", type=str, required=True, help="Path to litdata directory")
    parser.add_argument("--output", "--out", type=str, required=True, help="Path to output file")
    parser.add_argument("--bs", "--batch_size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument("--maplocation", "--maploc", type=str, default=None, help="Map location for model loading")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    model_path = args.em
    litdata_dir = args.litdata

    # Create output directory
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output if args.output.endswith(".h5") else f"{args.output}.h5"

    if not os.path.exists(output_path) or args.overwrite:
        print(f"Predicting with model {model_path} on data {litdata_dir}")
        predict(
            model_ckpt_path=model_path,
            litdata_dir=litdata_dir,
            output_path=output_path,
            map_location=args.map_location,
            batch_size=args.bs,
            save_h5=True,
        )
    else:
        print(f"Results already exist at {output_path}")
