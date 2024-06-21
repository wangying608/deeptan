import anndata
import scanpy as sc
import scvi


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import os
import sys


path_to_rna_anndata = "Chen-2019/Chen-2019-RNA.h5ad"
path_to_atac_anndata = "Chen-2019/Chen-2019-ATAC.h5ad"

adata_rna = anndata.read_h5ad(path_to_rna_anndata)
adata_atac = anndata.read_h5ad(path_to_atac_anndata)
adata_rna.var_names_make_unique()
adata_rna.obs_names_make_unique()
adata_atac.var_names_make_unique()
adata_atac.obs_names_make_unique()
# print(len(intersection(adata_atac.obs.index, adata_rna.obs.index)))


# Filter
sc.pp.filter_cells(adata_atac, min_genes=100)
sc.pp.filter_genes(adata_atac, min_cells=50)
sc.pp.filter_cells(adata_rna, min_genes=100)
sc.pp.filter_genes(adata_rna, min_cells=50)

# Find variable features
sc.pp.filter_genes_dispersion(adata_atac, n_top_genes=10000, n_bins=100)
sc.pp.filter_genes_dispersion(adata_rna, n_top_genes=10000, n_bins=100)


# PCA sc
sc.tl.pca(adata_atac, 6402)#6402
sc.tl.pca(adata_rna, 800)#3248

np.sum(adata_atac.uns['pca']['variance_ratio'])
np.sum(adata_rna.uns['pca']['variance_ratio'])


adata_atac.write(filename='anndata_scanpy_pca_atac.h5ad')
adata_rna.write(filename='anndata_scanpy_pca_rna.h5ad')
