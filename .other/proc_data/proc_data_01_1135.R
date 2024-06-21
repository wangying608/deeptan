# 
phen = read.table("/mnt/hdd2/data/data_xrn2p/processed/data_01_SFc3c343cB/1001genomes-FT10_FT16.tsv", header = TRUE, sep = "\t")
sras = read.table("/mnt/hdd2/data/1001/transcriptomes/GSE80744/SraRunTable.txt", header = TRUE, sep = ",")

in_6583_cultivars = sras$Cultivar
in_6583_experimen = sras$Experiment

in_6583_expe2cult = paste(in_6583_experimen, in_6583_cultivars, sep = "+")
in_6583_expe2cult.uniq = unique(in_6583_expe2cult)
expe2cult728 = strsplit(in_6583_expe2cult.uniq, "[+]") |> as.data.frame()

expe = c()
for (cult in 1:nrow(phen)) {
  if (phen$name[cult] %in% expe2cult728[2, ]) {
    expe = c(expe, expe2cult728[1, which(expe2cult728[2,]==phen$name[cult])])
  } else {
    expe = c(expe, NA)
  }
}

phen = cbind(expe, phen)

save.image("~/disks/hdd2/data/data_xrn2p/processed/data_01_SFc3c343cB/expe2phen.RData")
