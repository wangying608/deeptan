import argparse
import os

from deeptan.graph.recon import predict


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTAN prediction script.")
    parser.add_argument("--em", type=str, required=True, help="Existing model checkpoint path.")
    parser.add_argument("--litdata", "--data", type=str, required=True, help="Path to litdata directory")
    parser.add_argument("--output", "--out", type=str, required=True, help="Path to output pickle file")
    parser.add_argument("--maplocation", "--maploc", type=str, default=None, help="Map location for model loading")
    parser.add_argument("--getcor", action="store_true", help="Get correlations between feature pairs and labels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.em
    litdata_dir = args.litdata

    # Create output directory
    # output_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), "predicted_" + os.path.basename(os.path.dirname(model_path)))
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    # output_pkl_path = os.path.join(output_dir, f"{os.path.basename(os.path.dirname(litdata_dir))}_{os.path.basename(litdata_dir)}.pkl")
    output_pkl_path = args.output
    if not output_pkl_path.endswith(".pkl"):
        output_pkl_path += ".pkl"

    if (not os.path.exists(output_pkl_path) and not os.path.exists(output_pkl_path.replace(".pkl", ".h5"))) or args.overwrite:
        print(f"Predicting with model {model_path} on data {litdata_dir}")
        predict(
            model_ckpt_path=model_path,
            litdata_dir=litdata_dir,
            output_pkl_path=output_pkl_path,
            map_location=args.maplocation,
            batch_size=8,
            save_h5=True,
        )
    else:
        # print(f"Results already exist at {output_pkl_path}")
        if os.path.exists(output_pkl_path):
            print(f"Results already exist at {output_pkl_path}")
        else:
            print(f"Results already exist at {output_pkl_path.replace('.pkl', '.h5')}")

    # if args.getcor:
    #     output_cor_mat = os.path.join(os.path.dirname(output_pkl_path), os.path.basename(output_pkl_path) + "." + "correlation_matrix.npz")
    #     if not os.path.exists(output_cor_mat) or args.overwrite:
    #         compute_feature_correlations(output_cor_mat, output_pkl_path)
    #     else:
    #         print(f"Correlation matrix already exists at {output_cor_mat}")
