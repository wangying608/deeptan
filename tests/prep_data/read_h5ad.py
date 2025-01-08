r"""
Read h5ad file
"""
import os
import sys
import anndata


if __name__ == "__main__":
    path_h5ad = sys.argv[1]
    adata = anndata.read_h5ad(path_h5ad)
    print(adata)
