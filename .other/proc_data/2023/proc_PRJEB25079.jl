include("md_1_proc_rna.jl")
using JLD2


# getShMergeFilesPipe("/mnt/hdd2/data/PRJEB25079/SraRunTable.txt",
#     "/mnt/hdd2/data/PRJEB25079/00rawdata/",
#     "/mnt/hdd2/data/PRJEB25079/runs_cat/",
#     "/mnt/hdd2/data/PRJEB25079/cat.sh")


TPM_PRJEB25079 = concTPMFiles("")
TPM_PRJEB25079_f = delSmlRowInDf(TPM_PRJEB25079, 1.0, [1])

@save "TPM_PRJEB25079.jld2" {compress=true} TPM_PRJEB25079 TPM_PRJEB25079_f

CSV.write("TPM_PRJEB25079.csv", TPM_PRJEB25079)
CSV.write("TPM_PRJEB25079_f.csv", TPM_PRJEB25079_f)
