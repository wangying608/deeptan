# Where do 9190 cells come from?
import numpy as np
import pandas as pd
import scvi
import anndata


def intersection(lst1: list, lst2: list):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


path_rna_9190 = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-RNA.h5ad"
path_atac_9190 = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-ATAC.h5ad"

path_rna_bc = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/brainCortex_cDNA/cDNA.h5ad"
path_atac_bc = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/brainCortex_chromatin/chromatin.h5ad"

path_rna_b = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/adbrain_cDNA/cDNA.h5ad"
path_atac_b = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/adbrain_chromatin/chromatin.h5ad"


adata_rna = anndata.read_h5ad(path_rna_9190)
adata_atac = anndata.read_h5ad(path_atac_9190)

adata_rna_bc = anndata.read_h5ad(path_rna_bc)
adata_atac_bc = anndata.read_h5ad(path_atac_bc)

adata_rna_b = anndata.read_h5ad(path_rna_b)
adata_atac_b = anndata.read_h5ad(path_atac_b)


obs_T_rna = adata_rna.obs.T
obs_T_atac = adata_atac.obs.T

obs_T_rna_bc = adata_rna_bc.obs.T
obs_T_atac_bc = adata_atac_bc.obs.T

obs_T_rna_b = adata_rna_b.obs.T
obs_T_atac_b = adata_atac_b.obs.T


b_cell_rna_1 = list(obs_T_rna.columns)
b_cell_rna_2 = list(obs_T_rna_bc.columns)
b_cell_rna_3 = list(obs_T_rna_b.columns)
print(len(intersection(b_cell_rna_1, b_cell_rna_2)))#0
print(len(intersection(b_cell_rna_1, b_cell_rna_3)))# 9189: Chen-2019/adbrain_cDNA/cDNA.h5ad
print(len(intersection(b_cell_rna_2, b_cell_rna_3)))#0

b_cell_atac_1 = list(obs_T_atac.columns)
b_cell_atac_2 = list(obs_T_atac_bc.columns)
b_cell_atac_3 = list(obs_T_atac_b.columns)
print(len(intersection(b_cell_atac_1, b_cell_atac_2)))#0
print(len(intersection(b_cell_atac_1, b_cell_atac_3)))#0
print(len(intersection(b_cell_atac_2, b_cell_atac_3)))# 4755

# Really 0?
b_cell_atac_1_pure = [i.split("_")[-1] for i in b_cell_atac_1]
b_cell_atac_2_pure = [i.split("-")[0] for i in b_cell_atac_2]
b_cell_atac_3_pure = [i.split("_")[-1] for i in b_cell_atac_3]
print(len(intersection(b_cell_atac_1_pure, b_cell_atac_3_pure)))# 5475
print(len(intersection(b_cell_atac_1_pure, b_cell_atac_2_pure)))#1
print(len(intersection(b_cell_atac_2_pure, b_cell_atac_3_pure)))#1
