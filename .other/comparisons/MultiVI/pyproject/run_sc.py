import os

import numpy as np
import pandas as pd
import scvi
import anndata
import sys


prefix = sys.argv[1]
dir_save = "chen_9190/model_chen_9190" + "_r_" + prefix
path_latent_repr = os.path.join(dir_save, ("latent_repr_" + prefix + ".txt"))
print("  --- dir_save: " + dir_save)
print("  --- latent representation: " + path_latent_repr)


path_to_rna_anndata = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-RNA.h5ad"
path_to_atac_anndata = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-ATAC.h5ad"


adata_rna = anndata.read_h5ad(path_to_rna_anndata)
adata_atac = anndata.read_h5ad(path_to_atac_anndata)


# Gen multiomic adata
adata_multi = anndata.concat([adata_rna, adata_atac], axis=1)
#
# adata_multi.var.columns
# Index(['chrom', 'chromStart', 'chromEnd', 'genome', 'n_counts'], dtype='object')


adata_mvi = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)


scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
vae = scvi.model.MULTIVI(adata=adata_mvi, n_genes=28930, n_regions=241757)
vae.train()

scvi.model.MULTIVI.save(vae, dir_path=dir_save, prefix=prefix, save_anndata=False)

latent_repr = vae.get_latent_representation()

np.savetxt(path_latent_repr, latent_repr)

#

# model01 = scvi.model.MULTIVI.load(dir_path="/home/wuch/prjs/XRN2P/comparisons/MultiVI/pyproject/chen_9190/model_chen_9190_r_01",
#                                   adata=adata_mvi,
#                                   prefix='01')
#
# model02 = scvi.model.MULTIVI.load(dir_path="/home/wuch/prjs/XRN2P/comparisons/MultiVI/pyproject/chen_9190/model_chen_9190_r_02",
#                                   adata=adata_mvi,
#                                   prefix='02')

