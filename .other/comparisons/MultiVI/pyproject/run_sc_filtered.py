import numpy as np
import pandas as pd
import scvi
import anndata


def intersection(lst1: list, lst2: list):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# path_to_rna_anndata = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/adbrain_cDNA/cDNA.h5ad"
# path_to_atac_anndata = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/adbrain_chromatin/chromatin.h5ad"
# path_to_atac_anndata = "anndata_scanpy_pca_atac.h5ad"
# path_to_rna_anndata = "anndata_scanpy_pca_rna.h5ad"
path_to_atac_anndata = "Chen-2019_sklearn_PCA/anndata_pca_atac.h5ad"
path_to_rna_anndata = "Chen-2019_sklearn_PCA/anndata_pca_rna.h5ad"

adata_rna = anndata.read_h5ad(path_to_rna_anndata)
adata_atac = anndata.read_h5ad(path_to_atac_anndata)

adata_rna.var_names_make_unique()
adata_rna.obs_names_make_unique()
adata_atac.var_names_make_unique()
adata_atac.obs_names_make_unique()
print(len(intersection(adata_atac.obs.index, adata_rna.obs.index)))

# Gen multiomic adata
adata_multi = anndata.concat([adata_rna, adata_atac], axis=1, index_unique=None, join='inner')
adata_multi.var_names_make_unique()
adata_multi.obs_names_make_unique()
#

obs_uniq = list(np.unique(adata_multi.obs.index))
adata_rna_subset = adata_rna[obs_uniq, :]
adata_rna_subset.var_names_make_unique()
adata_rna_subset.obs_names_make_unique()


adata_mvi = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna_subset, adata_atac)


scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
vae = scvi.model.MULTIVI(adata=adata_mvi, n_genes=adata_rna.shape[1], n_regions=adata_atac.shape[1])
vae.train()

# scvi.model.MULTIVI.save(vae, dir_path="model_multivi_chen_filtered", save_anndata=True)
#
# latent_repr = vae.get_latent_representation()
#
# np.savetxt("latent_repr_filterd.txt", latent_repr)
