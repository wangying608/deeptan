include("md_1_proc_rna.jl")

folder = "/mnt/hdd1/data_xrn2p/origin/data_06_21gmagS9Ul/PRJEB32665/final_output/"
dfo = concExpFiles(folder)
CSV.write("/mnt/hdd1/data_xrn2p/processed/data_06_21gmagS9Ul/exp_PRJEB32665.csv", dfo)


dir_srx = "/mnt/hdd1/GSE80744/runrnaseq_SE/sln/"
count2TPMinDir(dir_srx)

tpm_out = concTPMFiles("/mnt/hdd1/GSE80744/TPM/")
CSV.write("/mnt/hdd1/GSE80744/TPM_GSE80744.csv", tpm_out)
