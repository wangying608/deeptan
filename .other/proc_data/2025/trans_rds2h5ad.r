library(sceasy)
library(reticulate)

# path_rds = "/mnt/bank/sc_sn/GSE270392/GSE270392_Gm_atlas_Root.rna.seurat.obj.rds"
# path_out = "/mnt/bank/sc_sn/GSE270392/GSE270392_Gm_atlas_Root.rna.h5ad"
argv <- commandArgs(trailingOnly = TRUE)
path_rds = argv[1]
path_out = argv[2]

# conda_env_name = "sceasy"
# use_condaenv(conda_env_name)

# reticulate::use_python("/home/wuch/miniforge3/envs/sceasy/bin/python")

loompy <- reticulate::import('loompy')

seurat_obj <- readRDS(path_rds)

sceasy::convertFormat(seurat_obj, from="seurat", to="anndata", outFile=path_out)
