include("md_1_proc_rna.jl")
using JLD2


# merge raw files
getShMergeFilesPipe("/mnt/hdd2/data/1001/transcriptomes/GSE54680/SraRunTable.txt",
    "/mnt/hdd2/data/1001/transcriptomes/GSE54680/runs/",
    "/mnt/hdd2/data/1001/transcriptomes/GSE54680/runs_cat/",
    "/mnt/hdd2/data/1001/transcriptomes/GSE54680/merge_GSE54680.sh")



# cat tpm

TPM_GSE54680 = concTPMFiles("/mnt/hdd2/data/1001/transcriptomes/GSE54680/topipe/final_output")

TPM_GSE54680_f = delSmlRowInDf(TPM_GSE54680, 1.0, [1])


@save "/mnt/hdd2/data/1001/transcriptomes/GSE54680/my/GSE54680_TPM.jld2" {compress=true} TPM_GSE54680 TPM_GSE54680_f

CSV.write("/mnt/hdd2/data/1001/transcriptomes/GSE54680/my/GSE54680_TPM_f1.csv", TPM_GSE54680_f)
CSV.write("/mnt/hdd2/data/1001/transcriptomes/GSE54680/my/GSE54680_TPM.csv", TPM_GSE54680)
