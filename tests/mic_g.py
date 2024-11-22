r"""
Read single-cell h5ad file and save it to a NPY file.
"""
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import h5py
import polars as pl


def read_h5ad(h5ad_file: str) -> anndata.AnnData:
    r"""Read h5ad file and return AnnData object.

    Args:
        h5ad_file (str): Path to h5ad file.

    Returns:
        anndata.AnnData: AnnData object.
    """
    adata = sc.read_h5ad(h5ad_file)
    return adata


def adata_to_npy(adata: anndata.AnnData, output_dir: str, output_prefix: str) -> None:
    r"""Save AnnData object to NPY files.

    Args:
        adata (anndata.AnnData): AnnData object.
        output_dir (str): Output directory.
        output_prefix (str): Output prefix.
    """
    X = adata.X.toarray()
    print(f"X shape: {X.shape}")
    # Transpose X
    X = X.T
    print(f"Transposed X shape: {X.shape}")
    return None
    # Fill missing values with 0
    X[np.isnan(X)] = 0.0
    X[np.isinf(X)] = 0.0
    X[np.isneginf(X)] = 0.0
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{output_prefix}_X.npy"), X)
    print(f"Saved X to {os.path.join(output_dir, f'{output_prefix}_X.npy')}")
    # obs = adata.obs.to_numpy()
    # print(f"obs shape: {obs.shape}")
    # np.save(os.path.join(output_dir, f"{output_prefix}_obs.npy"), obs)
    # print(f"Saved obs to {os.path.join(output_dir, f'{output_prefix}_obs.npy')}")


def adata_to_parquet(adata: anndata.AnnData, output_dir: str, output_prefix: str) -> None:
    r"""Save AnnData object to Parquet files.
    Args:
        adata (anndata.AnnData): AnnData object.
        output_dir (str): Output directory.
        output_prefix (str): Output prefix.
    """
    X = adata.X.toarray()
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    print(f"X shape: {X.shape}")

    os.makedirs(output_dir, exist_ok=True)

    # Create a Polars DataFrame with obs_names and var_names.
    obs_names = adata.obs_names.astype(str).to_list()
    var_names = adata.var_names.astype(str).to_list()
    df = pl.DataFrame({"obs_names": obs_names}).hstack(pl.DataFrame(X[:, :500], schema=var_names[:500]))
    print(f"DataFrame shape: {df.shape}")
    print(f"Head of DataFrame:\n{df.head()}\n")

    # Check number of None values
    print(f"Number of None values:\n{df.null_count().sum_horizontal()}\n")

    df.write_parquet(os.path.join(output_dir, f"{output_prefix}.parquet"))


if __name__ == "__main__":
    h5ad_file = sys.argv[1]
    adata = read_h5ad(h5ad_file)
    # adata_to_npy(adata, "data/test_mic_g_init", "scRNAseq")
    adata_to_parquet(adata, "data/test_mic_g_init", "scRNAseq")
    print("Done.")
