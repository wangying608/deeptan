include("md_1_proc_rna.jl")
using JLD2

# cat tpm

TPM_26 = concTPMFiles("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_26_Hk14v0x8Lu/rnaseq/final_output")

TPM_26_f = delSmlRowInDf(TPM_26, 1.0, [1])


@save "/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_26_Hk14v0x8Lu/rnaseq/my/26_TPM.jld2" {compress=true} TPM_26 TPM_26_f

CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_26_Hk14v0x8Lu/rnaseq/my/26_TPM_f1.csv", TPM_26_f)
CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_26_Hk14v0x8Lu/rnaseq/my/26_TPM.csv", TPM_26)
