include("md_1_proc_rna.jl")
using JLD2

# cat tpm

TPM_24 = concTPMFiles("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_24_qUFlpSlz95/rnaseq/final_output")

TPM_24_f = delSmlRowInDf(TPM_24, 1.0, [1])


@save "/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_24_qUFlpSlz95/rnaseq/my/24_TPM.jld2" {compress=true} TPM_24 TPM_24_f

CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_24_qUFlpSlz95/rnaseq/my/24_TPM_f1.csv", TPM_24_f)
CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_24_qUFlpSlz95/rnaseq/my/24_TPM.csv", TPM_24)
