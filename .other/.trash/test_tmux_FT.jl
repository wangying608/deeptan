# Test & Debug
cd("/home/wuch/a/new/test")

include("../src/md_1_PreNetInit.jl")

using JLD2, CSV, DataFrames

## Read data for testing

path_phen = "/home/wuch/a/new/data/FT_omics_3_feat_3000_sample_600/pheno_600.csv"
path_mrna = "/home/wuch/a/new/data/FT_omics_3_feat_3000_sample_600/mrna_1000.csv"
path_meth = "/home/wuch/a/new/data/FT_omics_3_feat_3000_sample_600/meth_3000.csv"
path_snps = "/home/wuch/a/new/data/FT_omics_3_feat_3000_sample_600/snp_1000.csv"

df_phen = CSV.read(path_phen, DataFrame; drop=[1])
df_mrna = CSV.read(path_mrna, DataFrame; drop=[1])
df_meth = CSV.read(path_meth, DataFrame; drop=[1])
df_snps = CSV.read(path_snps, DataFrame; drop=[1])


# mrna_meth = hcat(Matrix(mrna), Matrix(meth))
df1_phen = mapcols(myNormalize, df_phen)
df1_mrna = featureFilterSD(df_mrna, true, 0.5)
df1_meth = featureFilterSD(df_meth, true, 0.5)
df1_snps = featureFilterSD(df_snps, false, 0.001, 0.001)

@save "ft_dataset.jld2" {compress=true} df_phen df_meth df_mrna df_snps df1_phen df1_meth df1_mrna df1_snps
@load "ft_dataset.jld2"


# MI_mrna, samples_mrna = calcMI(df1_mrna)

# MI_methmrna, samples_methmrna = calcMI(df1_meth, df1_mrna)
MI_methphen, samples_methphen = calcMI(df1_meth, df1_phen)

# @save "MI_omics.jld2" {compress=true} MI_omics_mat samples_omics omics_mat


# TEST
tmp_sp = sortperm(MI_methphen[:])
fssort = names(df1_meth)[tmp_sp]
fssort[1:10] |> print
fssort[11:20] |> print
