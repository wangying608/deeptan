include("md_1_proc_rna.jl")


# merge raw files
getShMergeFilesPipe("/home/wuch/prjs/XRN2P/data_tmp/GSE43858/SraRunTable.txt",
    "/home/wuch/prjs/XRN2P/data_tmp/GSE43858/runs/00rawdata/",
    "/home/wuch/prjs/XRN2P/data_tmp/GSE43858/runs_cat/",
    "/home/wuch/prjs/XRN2P/data_tmp/GSE43858/cat_GSE43858.sh")

#
