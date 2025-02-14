import os
from deeptan.utils.uni import predict, process_results


if __name__ == "__main__":
    model_path = "/mnt/hdd1/wuch/logs/GSE155304_SRP273996/seed_42_20250212152331_170LZ/best-model-epoch=0003-val_loss=2.5736.ckpt"
    litdata_dir = "/mnt/hdd1/wuch/optimized_data/GSE155304_SRP273996/seed_42/tst"
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

    predict(
        model_ckpt_path=model_path,
        litdata_dir=litdata_dir,
        output_pickle_path=output_pickle_path,
        map_location=None,
        batch_size=4,
    )

    process_results(output_pickle_path, output_pkl)

    # with open(output_pkl, "rb") as f:
    #     results = pickle.load(f)
    # print(results.keys())
    # # For each key in the results dictionary, print data shape
    # for key in results.keys():
    #     print(f"Key: {key}, Shape: {results[key].shape}")
