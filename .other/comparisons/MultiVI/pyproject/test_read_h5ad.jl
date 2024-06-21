using Pkg
Pkg.activate("/home/wuch/.julia/environments/XRN2P.jl")
Pkg.instantiate()

using Muon
using DataFrames
using Statistics
# include("/home/wuch/prjs/XRN2P/src/function_tmapreduce.jl")
# using .TMapReduce


path_to_rna_anndata = "/home/wuch/prjs/XRN2P/comparisons/MultiVI/pyproject/Chen-2019/Chen-2019-RNA.h5ad"
path_to_atac_anndata = "/home/wuch/prjs/XRN2P/comparisons/MultiVI/pyproject/Chen-2019/Chen-2019-ATAC.h5ad"

file_rna = readh5ad(path_to_rna_anndata)
file_atac = readh5ad(path_to_atac_anndata)

n_region = size(file_atac)[2]

# std(file_atac.X[:, 997])
# atac_f997 = file_atac.X[:,997]
# atac_f997.n
# atac_f997.nzind
# atac_f997.nzval


# tSdVar(nthVar::Int64) = 
# tNumsVar(nthVar::Int64) = length(file_atac.X[:, nthVar].nzind)
# function tNumsVar(nthVar::Int64, Xspmat::Muon.SparseDataset{Float32}=file_atac.X)
#     nthLen = length(Xspmat[:, nthVar].nzind)
#     return nthLen
# end
# tNumsVar(997)
# nums_val_atac = tmapreduce(tNumsVar, vcat, 1:n_region)

nums_val_atac = zeros(Float32, n_region)
Threads.@threads for nvx in eachindex(nums_val_atac)
    nums_val_atac[nvx] = sum(view(file_atac.X, :, nvx))
end
