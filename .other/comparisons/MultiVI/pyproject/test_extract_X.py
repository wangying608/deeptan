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

print(type(adata_atac.X))
x_atac = adata_atac.X.astype(float)
