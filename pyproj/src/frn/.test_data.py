import os
from data_ncv import data_opt_trn, MyDataModule4Train, read_litdata_ncv_for_mi


if __name__ == "__main__":
    k_outer = 10
    k_inner = 5
    seed_split = 42
    output_dir = "/mnt/hdd2/homext/wuch/xn2p/test/data/proc_zma"
    paths_omics = [
        "/mnt/bank/CropGS-Hub/maize_1404/foruse/tpm_385_f.csv",
    ]
    path_label = "/mnt/bank/CropGS-Hub/maize_1404/foruse/maize_385_4_traits_fullnames.csv"
    out_dim = 1
    # traits_name = ["DTT", "PH", "KNPE", "KWPE"]
    traits_name = ["DTT"]

    optimized_data_dir = os.path.join(output_dir, "optimized")
    os.makedirs(optimized_data_dir, exist_ok=True)

    data_opt_trn(optimized_data_dir, paths_omics, path_label, out_dim, traits_name, k_outer, k_inner, seed_split, True, True)

    """
    datamodule = MyDataModule4Train(optimized_data_dir, k_outer, k_inner, 0, 0, 16)
    datamodule.setup()
    """

    # path_mi2g = "../../../mi2graph/target/x86_64-unknown-linux-musl/release/mi2graph"
    # read_litdata_ncv_for_mi(optimized_data_dir, output_dir, k_outer, k_inner, 0, 0, 100.0)
