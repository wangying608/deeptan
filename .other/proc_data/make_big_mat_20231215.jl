# Make the big matrix
using DataFrames, CSV
include("md_1_proc_rna.jl")


xdir = "/home/wuch/prjs/XRN2P/data_tmp/test_data_big1"
file_omics = ["atac_matrix_1215.csv", "data_06_ibaq_phos_rnaseq.csv", "exp.matrix.csv", "m6a_matrix2.tsv", "prot_matrix.csv"]

@doc """
Some omics types are allowed to be sparse.
"""
# Read files
# omics_dfs = map(x -> CSV.read(x, DataFrame; transpose=true), joinpath.(xdir, file_omics))
df_atac = CSV.read(joinpath(xdir, file_omics[1]), DataFrame;)[:,4:end]
df_exp = CSV.read(joinpath(xdir, file_omics[3]), DataFrame;)
df_m6A = CSV.read(joinpath(xdir, file_omics[4]), DataFrame; missingstring=["NA"])
df_prot = CSV.read(joinpath(xdir, file_omics[5]), DataFrame;)
# df_protRNA = CSV.read(joinpath(xdir, file_omics[2]), DataFrame;)

# Remove missings with %
df_atac = delSthRowInDf(df_atac, missing, Int64[1], 0.95)
df_exp = delSthRowInDf(df_exp, missing, Int64[1], 0.15)
df_m6A = delSthRowInDf(df_m6A, missing, Int64[1], 0.95)
df_prot = delSthRowInDf(df_prot, missing, Int64[1], 0.95)
# df_protRNA = delSthRowInDf(df_protRNA, missing, Int64[1], 0.95)

GC.gc()

# Fill missings with 0.0
df_atac = convMissing2Sth(df_atac, 0.0)
df_exp = convMissing2Sth(df_exp, 0.0)
df_m6A = convMissing2Sth(df_m6A, 0.0)
df_prot = convMissing2Sth(df_prot, 0.0)
# df_protRNA = convMissing2Sth(df_protRNA, 0.0)

# Remove 0.0 with %
df_atac = delSthRowInDf(df_atac, 0.0, Int64[1], 0.95)
df_exp = delSthRowInDf(df_exp, 0.0, Int64[1], 0.15)
df_m6A = delSthRowInDf(df_m6A, 0.0, Int64[1], 0.95)
df_prot = delSthRowInDf(df_prot, 0.0, Int64[1], 0.95)
# df_protRNA = delSthRowInDf(df_protRNA, 0.0, Int64[1], 0.95)

GC.gc()

println(size.([df_atac, df_exp, df_m6A, df_prot]))


@doc "Rename gene id"
#
tcols = "atac_" .* df_atac[:,1]
df_atac[:,1] = tcols
#
tcols_1 = "TPM_" .* df_exp[:,1]
df_exp[:,1] = tcols_1
#
tcols_2 = "m6A_" .* df_m6A[:,1]
df_m6A[:,1] = tcols_2
length(unique(df_m6A[:,1]))
#
