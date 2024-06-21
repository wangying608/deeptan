# 
library(foreach)

data_dir = "~/disks/hdd1/data_xrn2p/origin/data_06_21gmagS9Ul/PRJEB32665/final_output"
read_counts = read.table(paste0(data_dir, "/", "readcount.txt"), header = TRUE, sep = "\t")

cleanNameERR = function(dfmat){
  ERRs = colnames(dfmat)
  wh_cols = grep("ERR", ERRs)
  ERRs = ERRs[wh_cols]
  ERRs = foreach(coln = ERRs, .combine = "c") %do% {
    tmp = strsplit(coln, "[.]")[[1]]
    tmp = tmp[length(tmp)-1]
    tmp
  }
  new_df = dfmat
  colnames(new_df)[wh_cols] = ERRs
  return(new_df)
}

read_counts = cleanNameERR(read_counts)


files = dir(data_dir)
files = files[grep("exp.txt", files)]
f.t1 = read.table(paste0(data_dir, "/", files[17]), header = TRUE, sep = "\t")
