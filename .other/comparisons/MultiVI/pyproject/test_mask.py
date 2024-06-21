#
import os
import random
import numpy as np
import pandas as pd
import scvi
import anndata
import sys

# input arguments:
# prefix = sys.argv[1]
# mask_perc = float(sys.argv[2])
# seed_split = int(sys.argv[3])

# For features
mask_perc = 0.3
seed_mask = 1234
# For samples
# seed_split = 1234
trn_perc = 0.7
path_to_rna_anndata = "Chen-2019-RNA_test.h5ad"
path_to_atac_anndata = "Chen-2019-ATAC_test.h5ad"


def intersection(lst1: list, lst2: list):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# Read original data
adata_rna = anndata.read_h5ad(path_to_rna_anndata)
adata_atac = anndata.read_h5ad(path_to_atac_anndata)


# obs inter
obs_in = intersection(list(adata_rna.obs_names), list(adata_atac.obs_names))
n_obs = len(obs_in)
adata_atac = adata_atac[obs_in, :]
adata_rna = adata_rna[obs_in, :]


# Select table positions [x,y] to mask as zeros
# RNA
n_gene = adata_rna.shape[1]
n_elem_rna = n_obs * n_gene
random.seed(seed_mask)
posi2mask_rna = random.sample(range(0, n_elem_rna), int(mask_perc * n_elem_rna))
# ATAC
n_region = adata_atac.shape[1]
n_elem_atac = n_obs * n_region
random.seed(seed_mask)
posi2mask_atac = random.sample(range(0, n_elem_atac), int(mask_perc * n_elem_atac))
# Start mask
adata_atac_masked = adata_atac
adata_rna_masked = adata_rna
for xe1 in posi2mask_atac:
    adata_atac_masked.X[, ] = 0.0
for xe2 in posi2mask_rna:
    adata_rna_masked.X[, ] = 0.0


# Gen adata_multi
adata_multi = anndata.concat([adata_rna_masked, adata_atac_masked], axis=1)
adata_mvi = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna_masked, adata_atac_masked)
num_sample = adata_multi.shape[0]


# Split samples to trn and val
# random.seed(seed_split)
# trn = random.sample(range(0, num_sample), int(trn_perc * num_sample))
# val = [i for i in [j for j in range(0, num_sample)] if i not in trn]


# Run model
scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
vae = scvi.model.MULTIVI(adata=adata_mvi, n_genes=adata_rna_masked.shape[1], n_regions=adata_atac_masked.shape[1])
# vae.train(train_size=trn_perc)

# scvi.model.MULTIVI.save(vae, dir_path=dir_save, prefix=prefix, save_anndata=True)

# latent_repr = vae.get_latent_representation()

# np.savetxt(path_latent_repr, latent_repr)

