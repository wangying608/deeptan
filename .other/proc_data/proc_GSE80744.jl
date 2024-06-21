include("md_1_proc_rna.jl")
using JLD2


getShMergeFilesPipe("/mnt/hdd2/data/1001/transcriptomes/GSE80744/SraRunTable.txt",
    "/mnt/hdd2/data/1001/transcriptomes/GSE80744/sra/",
    "/mnt/hdd2/data/1001/transcriptomes/GSE80744/runs_cat/",
    "/mnt/hdd2/data/1001/transcriptomes/GSE80744/merge_GSE80744.sh")
#


TPM_GSE80744 = concTPMFiles("/mnt/hdd2/data/1001/transcriptomes/GSE80744/my/TPM_GSE80744")
TPM_GSE80744_f = delSmlRowInDf(TPM_GSE80744, 1.0, [1])

@save "/mnt/hdd2/data/1001/transcriptomes/GSE80744/my/TPM_GSE80744.jld2" {compress=true} TPM_GSE80744 TPM_GSE80744_f

CSV.write("/mnt/hdd2/data/1001/transcriptomes/GSE80744/my/TPM_GSE80744.csv", TPM_GSE80744)
CSV.write("/mnt/hdd2/data/1001/transcriptomes/GSE80744/my/TPM_GSE80744_f.csv", TPM_GSE80744_f)

# @load "/mnt/hdd2/data/1001/transcriptomes/GSE80744/my/TPM_GSE80744.jld2"

