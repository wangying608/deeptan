import os
from deeptan.utils.data import (
    read_h5ad,
    h5ad_to_parquet,
    h5_to_parquet,
    h5mu_to_parquet,
    split_parquet,
    split_parquet_with_celltypes,
    split_parquet_with_joint_strata,
)


if __name__ == "__main__":
    # path_h5mu = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control.h5mu"
    # h5mu_to_parquet(path_h5mu, path_h5mu.replace(".h5mu", ""))

    # path_parq = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control.parquet"
    # split_parquet(path_parq, path_parq.replace(".parquet", "")+"_split", [0.8, 0.1, 0.1], [i+42 for i in range(5)])

    # h5ad_to_parquet("/home/wuch/Downloads", "/home/wuch/Downloads")

    path_h5ad = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scRNA/SRP273996.h5ad"
    assert path_h5ad.endswith(".h5ad")
    path_parquet = path_h5ad.replace(".h5ad", ".parquet")
    if not os.path.exists(path_parquet):
        h5ad_to_parquet(path_h5ad, path_parquet, True)

    celltypes = read_h5ad(path_h5ad).obs["Celltype"].to_list()
    batch_id = read_h5ad(path_h5ad).obs["Orig.ident"].to_list()

    output_dir = path_parquet.replace(".parquet", "") + "_split" + "_resampled"
    ratio = [0.8, 0.1, 0.1]
    seeds = [i + 42 for i in range(5)]
    # split_parquet_with_celltypes(
    #     cell_types=celltypes,
    #     parquet_file=path_parquet,
    #     output_dir=output_dir,
    #     ratio=ratio,
    #     seeds=seeds,
    # )
    split_parquet_with_joint_strata(
        cell_types=celltypes,
        orig_idents=batch_id,
        parquet_file=path_parquet,
        output_dir=output_dir,
        ratio=ratio,
        seeds=seeds,
        balance_strategy="combined",
    )


# Example usage
# if __name__ == "__main__":
#     # Load your data
#     cell_types = [...]  # List of cell type labels
#     orig_idents = [...]  # List of origin identifiers
#     split_parquet_with_joint_strata(
#         cell_types=cell_types,
#         orig_idents=orig_idents,
#         parquet_file="input.parquet",
#         output_dir="output_splits",
#         ratio=[0.8, 0.1, 0.1],
#         seeds=[42, 43, 44],
#         allow_duplicates=True
#     )
