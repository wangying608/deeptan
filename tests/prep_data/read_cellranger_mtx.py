r"""
Read a cellranger mtx directory.
"""
# import os
import sys
import scanpy
# import polars as pl


def read_cellranger_mtx(dir_mtx: str) -> scanpy.AnnData:
    r"""
    Read a cellranger mtx directory.

    Args:
        dir_mtx: Directory containing the cellranger mtx files.
    """
    adata = scanpy.read_10x_mtx(dir_mtx, gex_only=False, make_unique=False)
    adata.var_names_make_unique(join="_")

    print(f"n_obs: {adata.n_obs}")
    print(f"n_vars: {adata.n_vars}")
    print(f"Feature types:\n{adata.var["feature_types"].unique()}\n")
    # print(f"Features 0 to 4:\n{adata.var[:5]}\n")
    # print(f"Features -1 to -5:\n{adata.var[-5:]}\n")

    # Filter RNA features
    var_gex_ind = adata.var["feature_types"].values == "Gene Expression"
    print(f"Max value of GEX: {adata[:, var_gex_ind].X.max()}")
    print(f"Min value of GEX: {adata[:, var_gex_ind].X.min()}")
    # adata = adata[adata[:, var_gex_ind].X.sum(axis=1) > 1000, :]
    # adata = adata[adata[:, var_gex_ind].X.sum(axis=1) < 25000, :]

    # Filter ATAC features
    var_atac_ind = adata.var["feature_types"].values == "Peaks"
    print(f"Max value of ATAC: {adata[:, var_atac_ind].X.max()}")
    print(f"Min value of ATAC: {adata[:, var_atac_ind].X.min()}")
    # adata = adata[adata[:, var_atac_ind].X.sum(axis=1) > 1000, :]
    # adata = adata[adata[:, var_atac_ind].X.sum(axis=1) < 100000, :]
    
    # Delete vars with zero counts
    adata = adata[:, adata.X.sum(axis=0) > 0]
    print(f"Matrix shape after delete vars with zero counts: {adata.shape}")

    return adata


if __name__ == "__main__":
    dir_mtx = sys.argv[1]
    
    adata = read_cellranger_mtx(dir_mtx)
    print(f"n_obs: {adata.n_obs}")
    print(f"n_vars: {adata.n_vars}")
