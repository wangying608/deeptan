include("md_1_proc_rna.jl")
using JLD2

# cat tpm

TPM_07 = concTPMFiles("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_07_kzIBeJwfuv/PRJNA899318/final_output")

TPM_07_f = delSmlRowInDf(TPM_07, 1.0, [1])


@save "/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_07_kzIBeJwfuv/PRJNA899318/my/07_TPM.jld2" {compress=true} TPM_07 TPM_07_f

CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_07_kzIBeJwfuv/PRJNA899318/my/07_TPM_f1.csv", TPM_07_f)
CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_07_kzIBeJwfuv/PRJNA899318/my/07_TPM.csv", TPM_07)
