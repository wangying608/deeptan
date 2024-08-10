include("md_1_proc_rna.jl")
using JLD2


TPM_PRJNA433874 = concTPMFiles("/mnt/hdd2/data/data_xrn2p/origin/data_13_H7hN0lPCP4/GSE110539/PRJNA433874/final_output")
TPM_PRJNA433874_f = delSmlRowInDf(TPM_PRJNA433874, 1.0, [1])

@save "/mnt/hdd2/data/data_xrn2p/origin/data_13_H7hN0lPCP4/GSE110539/PRJNA433874/my/TPM_PRJNA433874.jld2" {compress=true} TPM_PRJNA433874 TPM_PRJNA433874_f

CSV.write("/mnt/hdd2/data/data_xrn2p/origin/data_13_H7hN0lPCP4/GSE110539/PRJNA433874/my/TPM_PRJNA433874.csv", TPM_PRJNA433874)
CSV.write("/mnt/hdd2/data/data_xrn2p/origin/data_13_H7hN0lPCP4/GSE110539/PRJNA433874/my/TPM_PRJNA433874_f.csv", TPM_PRJNA433874_f)
