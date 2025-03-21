import argparse
import os

from deeptan.graph.recon import compute_feature_correlations, predict, process_results


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTAN prediction script.")
    parser.add_argument("--em", type=str, required=True, help="Existing model checkpoint path.")
    parser.add_argument("--litdata", "--data", type=str, required=True, help="Path to litdata directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.em
    litdata_dir = args.litdata
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(model_path)),
        "predicted_" + os.path.basename(os.path.dirname(model_path)),
    )
    os.makedirs(output_dir, exist_ok=True)
    output_pickle_path = os.path.join(
        output_dir,
        f"{os.path.basename(os.path.dirname(litdata_dir))}_{os.path.basename(litdata_dir)}.pkl",
    )
    output_pkl = output_pickle_path.replace(".pkl", "_numpy.pkl")

    # predict(
    #     model_ckpt_path=model_path,
    #     litdata_dir=litdata_dir,
    #     output_pickle_path=output_pickle_path,
    #     map_location=None,
    #     batch_size=4,
    # )

    # process_results(output_pickle_path, output_pkl)

    compute_feature_correlations(output_pkl)

    # with open(output_pkl, "rb") as f:
    #     results = pickle.load(f)
    # print(results.keys())
    # # For each key in the results dictionary, print data shape
    # for key in results.keys():
    #     print(f"Key: {key}, Shape: {results[key].shape}")
