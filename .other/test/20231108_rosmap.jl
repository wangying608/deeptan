# 
include("../src/act_env.jl")
include("../src/md_1_PreNetInit.jl")
using JLD2, CSV, DataFrames

path_rosmap_features = "/home/wuch/Downloads/ROSMAP.csv"
rosmap_features = CSV.read(path_rosmap_features, DataFrame, drop=[1])

# rosmap_features = featureFilterSD(rosmap_features)
normEachCol!(rosmap_features)

rosmap_features_mat = Matrix(rosmap_features)

outs = calcMIsLinearly(rosmap_features_mat)

@show findmax(outs[:, 1])
@show findmin(outs[:, 1])


using CairoMakie
hist(outs[:,1]; bins=100, figure = (; resolution = (1200, 800)))

# Restore matrix
# using Combinatorics: combinations
mat_restored = zeros(Float64, size(rosmap_features_mat, 2), size(rosmap_features_mat, 2))
indices_iter = combinations(1:size(rosmap_features_mat, 2), 2)
itn = 1
for itx in indices_iter
    mat_restored[itx[1], itx[2]] = outs[itn, 1]
    itn = itn + 1
end
itn = 1
for itx in indices_iter
    mat_restored[itx[2], itx[1]] = outs[itn, 1]
    itn = itn + 1
end
for nfeat in 1:size(rosmap_features_mat, 2)
    mat_restored[nfeat, nfeat] = atomMI(rosmap_features_mat[:, nfeat], rosmap_features_mat[:, nfeat], 0.02, [0.6, 0.99], 0.02)[1,1]
end

CSV.write("/home/wuch/prjs/XRN2P/frn/rosmap_mi_20231110_1119.csv", Tables.table(mat_restored))
