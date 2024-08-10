include("md_1_proc_rna.jl")
using JLD2

# cat tpm

TPM_25 = concTPMFiles("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_25_zsjaXmvq3M/rnaseq/final_output")

TPM_25_f = delSmlRowInDf(TPM_25, 1.0, [1])


@save "/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_25_zsjaXmvq3M/rnaseq/my/25_TPM.jld2" {compress=true} TPM_25 TPM_25_f

CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_25_zsjaXmvq3M/rnaseq/my/25_TPM_f1.csv", TPM_25_f)
CSV.write("/home/wuch/disks/hdd2/data/data_xrn2p/origin/data_25_zsjaXmvq3M/rnaseq/my/25_TPM.csv", TPM_25)
