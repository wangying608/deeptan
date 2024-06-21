include("md_1_proc_rna.jl")
# using CSV, DataFrames
using JLD2

folder0 = "/mnt/hdd2/data/1001/gene_body_methylation_data/"
path_ids = joinpath(folder0, "id_name.txt")
path_meth_mcg = joinpath(folder0, "Araport11_GB_mCG_strict.tsv")
path_meth_mchg = joinpath(folder0, "Araport11_GB_mCHG_strict.tsv")
path_meth_mchh = joinpath(folder0, "Araport11_GB_mCHH_strict.tsv")

ara11_ids = CSV.read(path_ids, DataFrame)[1:end-2, :]
ara11_mcg = CSV.read(path_meth_mcg, DataFrame, header=false, missingstring="NA")
ara11_mchg = CSV.read(path_meth_mchg, DataFrame, header=false, missingstring="NA")
ara11_mchh = CSV.read(path_meth_mchh, DataFrame, header=false, missingstring="NA")

ara11_ids = delSthRowInDf(ara11_ids, missing)
ara11_mcg = delSthRowInDf(ara11_mcg, missing, [1])
ara11_mchg = delSthRowInDf(ara11_mchg, missing, [1])
ara11_mchh = delSthRowInDf(ara11_mchh, missing, [1])

# set colnames
# colnames0 = vcat(["Gene_ID"], string.("EcoID_", ara11_ids[:,1])) |> unique
# rename!(ara11_mcg, colnames0)

# @save "/mnt/hdd2/data/1001/gene_body_methylation_data/Araport11_GB_mXXX_strict.JLD2" {compress=true} ara11_ids ara11_mcg ara11_mchg ara11_mchh

@load joinpath(folder0, "Araport11_GB_mXXX_strict.JLD2")

CSV.write(joinpath(folder0, "f_ids.csv"), ara11_ids)
CSV.write(joinpath(folder0, "f_mCG.csv"), ara11_mcg)
CSV.write(joinpath(folder0, "f_mCHG.csv"), ara11_mchg)
CSV.write(joinpath(folder0, "f_mCHH.csv"), ara11_mchh)

# tmp_test = dropmissing(ara11_mcg)

# xmax = maximum(eachrow(dropmissing(mcg_f[:, 2:end]))) |> maximum
# xmin = minimum(eachrow(dropmissing(mcg_f[:, 2:end]))) |> minimum
# any(x -> ismissing(x), unique([missing, 12, 9.0, missing]))

mcg_f = convMissing2Sth(ara11_mcg, 0.0) |> dropmissing
mcg_f = delSthRowInDf(mcg_f, 0.0, [1])

# 选取1107个样本内能用的样本后，再筛特征。
