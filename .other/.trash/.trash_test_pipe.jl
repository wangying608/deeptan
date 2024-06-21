# Trashed test of pipeline


cor2m_pearson = abs.(cor(Matrix(mrna), Matrix(meth)))
cor2m_spearman = abs.(corspearman(Matrix(mrna), Matrix(meth)))
cor2m_x123 = cor_autox123(Matrix(mrna), Matrix(meth), true)

pvalue(VarianceFTest([1,2,3], [4,5,6]))

## OR
CSV.write("cor_myx123_2omics.csv", Tables.table(cor2m_x123))
## CSV file is much larger.


# hclust in the same omics
cor_mrna_x123 = cor_autox123(Matrix(mrna), Matrix(mrna))
hclust(cor_mrna_x123)



#-------------------- 20230515 --------------------#


# Test & Debug
cd("/home/wuch/a/new/test")

include("../src/md_1_PreNetInit.jl")
# using .PreNetInit

using CairoMakie
using JLD2, CSV, DataFrames


## Read data for testing

path_mrna = "/home/wuch/a/new/data/FT_omics_3_feat_3000_sample_600/mrna_1000.csv"
mrna = CSV.read(path_mrna, DataFrame; drop=[1])

path_meth = "/home/wuch/a/new/data/FT_omics_3_feat_3000_sample_600/meth_3000.csv"
meth = CSV.read(path_meth, DataFrame; drop=[1])


## Normalization / Standardization
# mapcols!(myNormalize, mrna)
# mapcols!(myNormalize, meth)

# omics_mat = hcat(mrna, meth)
omics_mat = hcat(Matrix(mrna), Matrix(meth))

@save "omics_mat.jld2" {compress=true} omics_mat

@load "omics_mat.jld2"


#----------------------------------------------------

# x1, y1 = rmOutliersVecXY(omics_mat[:, 1], omics_mat[:, 2])
# tst_MI = optimLocalMI(x1, y1, 0.6, 0.02, getOnlyOptim=false)
# #
# scatter(x1, y1; color=(:blue, 0.3),
#     figure = (; resolution = (1080, 1080)))
# #
# scatter(x1[tst_MI.local_samples], y1[tst_MI.local_samples]; color=(:blue, 0.3),
#     figure = (; resolution = (1080, 1080)))
# #
spe = sortperm(mrna[:,78])
x78 = mrna[:,78][spe]
x79 = mrna[:,79][spe]

MIxy(x78[1:360], x79[1:360])

findall(iszero, x78)
findall(iszero, rmOutliers(x78))

std(x78)
std(x79)

fsmrna = featureFilterSD(mrna, 0.12)
names(mrna)[78]
fsmrna[!, names(mrna)[78]]

length(mrna[:,78])
std(mrna[:,78])
rec_sd = slidingWindow1D(std, false, sort(mrna[:,78]), 1, [0.6,0.98], 0.01)
any(x -> x < 0.11, rec_sd[:, "results"])
# hist(rec_sd[1:241, "results"]; bins=10)
# std(rec_sd[1:241, "results"])
std(sort(mrna[:, 78])[1:500])

optimum, whSamp = slidingWindow2D(MIxy, true, mrna[:,1], mrna[:,298], 1, [0.6,0.98], 0.01)
findmax(dfx[!, "results"])
dfx[204,:]
hexbin(mrna[:,1], mrna[:,298];
        bins=75,
        cellsize=0.05,
        colormap = :heat,
        figure = (; resolution = (1080, 1080)))
#
hexbin(mrna[91:600, 1], mrna[91:600, 298];
        bins=75,
        cellsize=0.05,
        colormap = :heat,
        figure = (; resolution = (1080, 1080)))

#----------------------------------------------------


## MI (quantifies the "amount of information") (larger is "better")
# MI_upTri = calcMITriangle(omics_mat)
MI_omics_mat = calcMI(omics_mat)


# MI_mrna = calcMITriangle(Matrix(mrna)[:, 78:79])
# tmpmrna = mrna[:, 78:79]
# CSV.write("tmp.csv", tmpmrna)

# MI_omics_vec = filter(!isnan, [x::Float64 for x in MI_omics_mat])

@save "MI_omics.jld2" {compress=true} MI_omics_mat

@load "MI_omics.jld2"


# hclust(cor2m)
# mcl(cor2m)
# affinityprop(cor2m)

