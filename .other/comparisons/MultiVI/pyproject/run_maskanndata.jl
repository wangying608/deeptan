include("md_MaskAnnData.jl")
using .MaskAnnData
using JLD2


path_to_rna_anndata = "/home/wuch/prjs/XRN2P/comparisons/MultiVI/pyproject/Chen-2019/Chen-2019-RNA.h5ad"
# path_to_rna_anndata = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-RNA.h5ad"

path_to_atac_anndata = "/home/wuch/prjs/XRN2P/comparisons/MultiVI/pyproject/Chen-2019/Chen-2019-ATAC.h5ad"
# path_to_atac_anndata = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-ATAC.h5ad"


adRNA, adATAC = maskAnndata(path_to_rna_anndata, path_to_atac_anndata, 0.1, 1234)
@save "ad_masked_01_seed_1234.jld2" {compress=true} adRNA adATAC
