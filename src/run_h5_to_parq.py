from deeptan.utils.data import h5_to_parquet, h5mu_to_parquet, split_parquet


if __name__ == "__main__":
    # path_h5mu = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control.h5mu"
    # h5mu_to_parquet(path_h5mu, path_h5mu.replace(".h5mu", ""))

    path_parq = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control.parquet"
    split_parquet(path_parq, path_parq.replace(".parquet", "")+"_split", [0.8, 0.1, 0.1], [i+42 for i in range(10)])
