import os

import numpy as np
import pandas as pd
import scvi
import anndata
import sys


prefix = sys.argv[1]
dir_save = "" + "_r_" + prefix
path_latent_repr = os.path.join(dir_save, ("latent_repr_" + prefix + ".txt"))
print("  --- dir_save: " + dir_save)
print("  --- latent representation: " + path_latent_repr)


path_to_rna_anndata = ""
path_to_atac_anndata = ""


adata_rna = anndata.read_h5ad(path_to_rna_anndata)
adata_atac = anndata.read_h5ad(path_to_atac_anndata)


# Gen multiomic adata
adata_multi = anndata.concat([adata_rna, adata_atac], axis=1)

# adata_multi.var.columns
# Index(['chrom', 'chromStart', 'chromEnd', 'genome', 'n_counts'], dtype='object')


adata_mvi = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)


scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
vae = scvi.model.MULTIVI(adata=adata_mvi, n_genes=adata_rna.shape[1], n_regions=adata_atac.shape[1])
vae.train()

scvi.model.MULTIVI.save(vae, dir_path=dir_save, prefix=prefix, save_anndata=False)

latent_repr = vae.get_latent_representation()

np.savetxt(path_latent_repr, latent_repr)

