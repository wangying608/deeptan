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


def intersection(lst1: list, lst2: list):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


path_to_rna_anndata = "Chen-2019/Chen-2019-RNA.h5ad"
path_to_atac_anndata = "Chen-2019/Chen-2019-ATAC.h5ad"

adata_rna = anndata.read_h5ad(path_to_rna_anndata)
adata_atac = anndata.read_h5ad(path_to_atac_anndata)
adata_rna.var_names_make_unique()
adata_rna.obs_names_make_unique()
adata_atac.var_names_make_unique()
adata_atac.obs_names_make_unique()
print(len(intersection(adata_atac.obs.index, adata_rna.obs.index)))


# Filter
sc.pp.filter_cells(adata_atac, min_genes=100)
sc.pp.filter_genes(adata_atac, min_cells=50)
sc.pp.filter_cells(adata_rna, min_genes=100)
sc.pp.filter_genes(adata_rna, min_cells=50)

# Find variable features
sc.pp.filter_genes_dispersion(adata_atac, n_top_genes=10000, n_bins=100)
sc.pp.filter_genes_dispersion(adata_rna, n_top_genes=10000, n_bins=100)


# PCA sc
# sc.tl.pca(adata_atac, 100)
# sc.tl.pca(adata_rna, 100)

# sklearn PCA
pca_atac = PCA(n_components=0.99)
pca_atac.fit(adata_atac.X.H.A)
print(pca_atac.explained_variance_ratio_.sum())
pca_rna = PCA(n_components=0.99)
pca_rna.fit(adata_rna.X.H.A)
print(pca_rna.explained_variance_ratio_.sum())

# sklearn SVD
# svd = TruncatedSVD(n_components=100, n_iter=100)
# svd.fit(adata_atac.X)
# print(svd.explained_variance_ratio_.sum())

# UMAP
# sc.pp.neighbors(adata_atac)
# sc.tl.umap(adata_atac, n_components=100)
# sc.pp.neighbors(adata_rna)
# sc.tl.umap(adata_rna, n_components=100)


# Plot
# sc.set_figure_params(dpi=300, dpi_save=300, format='png', figsize=(10, 8))
# legend_locations = ['on data', 'right margin']
# legend_location = legend_locations[1]
# #
# sc.pl.pca_variance_ratio(adata_atac)
# sc.pl.pca_variance_ratio(adata_rna)
# sc.pl.pca(adata_atac, color=['cell_type'], title='Cell type', legend_loc=legend_location)
# sc.pl.pca(adata_rna, color=['cell_type'], title='Cell type', legend_loc=legend_location)
# sc.pl.umap(adata_atac, color=['cell_type'], title='Cell type', legend_loc=legend_location)
# sc.pl.umap(adata_rna, color=['cell_type'], title='Cell type', legend_loc=legend_location)
#
#
# total_variance_atac = sum(adata_atac.uns['pca']['variance'])
# total_variance_rna = sum(adata_rna.uns['pca']['variance'])


# Generate anndata objects
var_atac = pd.DataFrame(data=[i for i in range(1, 1 + pca_atac.n_components_)],
                        index=[('pc_atac_' + str(i)) for i in range(1, 1 + pca_atac.n_components_)],
                        columns=['id'])

var_rna = pd.DataFrame(data=[i for i in range(1, 1 + pca_rna.n_components_)],
                       index=[('pc_rna_' + str(i)) for i in range(1, 1 + pca_rna.n_components_)],
                       columns=['id'])

adata_atac_pca = sc.AnnData(normalize(np.transpose(pca_atac.components_), axis=0), obs=adata_atac.obs, var=var_atac)
adata_rna_pca = sc.AnnData(normalize(np.transpose(pca_rna.components_), axis=0), obs=adata_rna.obs, var=var_rna)

adata_atac_pca.write(filename='anndata_pca_atac.h5ad')
adata_rna_pca.write(filename='anndata_pca_rna.h5ad')
