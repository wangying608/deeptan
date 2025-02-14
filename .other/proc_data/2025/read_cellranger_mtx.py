r"""
Read a cellranger mtx directory or h5 file.
"""

# import os
import sys

# import polars as pl
import anndata
import scanpy
import muon
import mudata


def read_cellranger_mtx_or_h5(dir_mtx_or_h5: str) -> scanpy.AnnData:
    r"""
    Read a cellranger mtx directory or h5 file.

    Args:
        dir_mtx_or_h5: Directory containing the mtx files or the path to h5 file.
    """
    # adata = scanpy.read_10x_mtx(dir_mtx, gex_only=False, make_unique=False)
    if dir_mtx_or_h5.endswith(".h5"):
        adata = scanpy.read_10x_h5(dir_mtx_or_h5, gex_only=False)
    else:
        adata = scanpy.read_10x_mtx(dir_mtx_or_h5, gex_only=False, make_unique=False)

    adata.var_names_make_unique(join="_")

    print(f"Matrix shape: {adata.shape}")
    print(f"n_obs: {adata.n_obs}")
    print(f"n_vars: {adata.n_vars}")
    print(f"Feature types: {adata.var['feature_types'].unique()}\n")
    print(f"Features 0 to 4:\n{adata.var[:5]}\n")
    print(f"Features -1 to -5:\n{adata.var[-5:]}\n")
    print(f"First 5 obs names:\n{adata.obs_names[:5]}\n")

    # Print properties
    print(f"\nobs properties:\n{adata.obs.keys()}")
    print(f"\nvar properties:\n{adata.var.keys()}")

    # Print first 5 obs counts
    # print(f"First 5 obs counts:\n{adata[:5].X}")
    # Print sum of first 5 obs counts
    # print(f"Sum of first 5 obs counts:\n{adata[:5].X.sum(axis=1)}")
    print("Median of counts per cell:")
    print(f"{adata.to_df().sum(axis=1).median()}")
    print("Mean of counts per cell:")
    print(f"{adata.to_df().sum(axis=1).mean()}")

    # Filter RNA features
    var_gex_ind = adata.var["feature_types"].values == "Gene Expression"
    print(f"\nNumber of GEX features: {var_gex_ind.sum()}")
    print(f"Max value of GEX: {adata[:, var_gex_ind].X.max()}")
    print(f"Min value of GEX: {adata[:, var_gex_ind].X.min()}")
    # adata = adata[adata[:, var_gex_ind].X.sum(axis=1) > 1000, :]
    # adata = adata[adata[:, var_gex_ind].X.sum(axis=1) < 25000, :]

    # Filter ATAC features
    var_atac_ind = adata.var["feature_types"].values == "Peaks"
    print(f"\nNumber of ATAC features: {var_atac_ind.sum()}")
    print(f"Max value of ATAC: {adata[:, var_atac_ind].X.max()}")
    print(f"Min value of ATAC: {adata[:, var_atac_ind].X.min()}")
    # adata = adata[adata[:, var_atac_ind].X.sum(axis=1) > 1000, :]
    # adata = adata[adata[:, var_atac_ind].X.sum(axis=1) < 100000, :]

    # Delete vars with zero counts
    adata = adata[:, adata.X.sum(axis=0) > 0]
    print(f"\nMatrix shape after delete vars with zero counts: {adata.shape}")
    print(
        f"Number of GEX features after delete vars with zero counts: {(adata.var['feature_types'].values == 'Gene Expression').sum()}"
    )
    print(
        f"Number of ATAC features after delete vars with zero counts: {(adata.var['feature_types'].values == 'Peaks').sum()}"
    )

    return adata


if __name__ == "__main__":
    file_path = sys.argv[1]
    adata = read_cellranger_mtx_or_h5(file_path)

    # # Save obs names to a parquet file
    # obs_names = adata.obs_names.to_list()
    # df_obs_names = pl.DataFrame({"obs_names": obs_names})
    # # Save to ~/Downloads/xxxx_obs_names.parquet
    # df_obs_names.write_parquet(os.path.join(os.path.expanduser("~"), "Downloads", f"{os.path.basename(dir_mtx)}_obs_names.parquet"))

    # h5_rep1 = "/mnt/bank/sc_sn/GSE235510/GSE235510_control_rep1/outs/filtered_feature_bc_matrix.h5"
    # h5_rep2 = "/mnt/bank/sc_sn/GSE235510/GSE235510_control_rep2/outs/filtered_feature_bc_matrix.h5"

    # adata_rep1 = read_cellranger_mtx_or_h5(h5_rep1)
    # print("\n--------------------------------------\n")
    # adata_rep2 = read_cellranger_mtx_or_h5(h5_rep2)

    # # adata_rep1_rep2 = concat([adata_rep1, adata_rep2], axis=1)
    # # print(adata_rep1_rep2.var["feature_types"].value_counts())

    # print("\n--------------------------------------\n")

    # # Union features of adata_rep1 and adata_rep2

    # # adata_rep1_rep2 = anndata.concat(
    # #     [adata_rep1, adata_rep2],
    # #     join="outer",
    # #     merge="first",
    # #     uns_merge="first",
    # #     label="replicate",
    # #     keys=["rep1", "rep2"],
    # #     index_unique="_",
    # # )
    # adata_rep1_rep2 = mudata.concat({"rep1": adata_rep1, "rep2": adata_rep2}, join="outer", label="replicate", index_unique="_")

    # # adata_rep1_rep2.obs_names_make_unique(join="_")
    # print("\nUnion features of adata_rep1 and adata_rep2:")
    # print(adata_rep1_rep2.shape)
    # # Print the feature types
    # print("Feature types in adata_rep1_rep2:")
    # print(adata_rep1_rep2.var["feature_types"].value_counts())

    # # Save the concatenated data to an h5 file
    # # scanpy.write(
    # #     filename="/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control.h5",
    # #     adata=adata_rep1_rep2,
    # #     ext="h5",
    # # )

    # # Convert the concatenated data to muon object
    # muon_obj = muon.MuData(adata_rep1_rep2)
    # print("\nMuData object created from concatenated AnnData:")
    # print(muon_obj)
