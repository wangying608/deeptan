import numpy as np
import pandas as pd
import scvi
import anndata


path_to_rna_anndata = "/mnt/hdd1/data_mv/GSE155304/rnaseq.h5ad"
path_to_atac_anndata = "/mnt/hdd1/data_mv/GSE155304/chromatin.h5ad"


adata_rna = anndata.read_h5ad(path_to_rna_anndata)
adata_atac = anndata.read_h5ad(path_to_atac_anndata)

adata_rna.obs_names = [i.split("_")[0] for i in adata_rna.obs_names]

adata_rna.obs_names_make_unique()
adata_rna.var_names_make_unique()

# Gen multiomic adata
adata_multi = anndata.concat([adata_rna, adata_atac], axis=1)
#


adata_mvi = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)


scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
vae = scvi.model.MULTIVI(adata=adata_mvi, n_genes=28930, n_regions=241757)
vae.train()

scvi.model.MULTIVI.save(vae, dir_path="model_multivi_chen", save_anndata=True)
# vae.save("test01")
latent_repr = vae.get_latent_representation()

np.savetxt("latent_repr.txt", latent_repr)
