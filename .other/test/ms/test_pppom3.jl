# Test & Debug
path_ppp = "/home/wuch/disks/hdd1/data_xrn2p/processed/data_06_21gmagS9Ul/ppp_all.tsv"

cd("/home/wuch/prjs/XRN2P/test/ms")

include("../../src/act_env.jl")

# include("../../src/md_1_PreNetInit.jl")
include("../../src/md_1_PreNetInit.jl")

using JLD2, CSV, DataFrames
using Profile, PProf


## Read data for testing
println("Reading the table")

df_ppp = CSV.read(path_ppp, DataFrame)#; drop=[1]
# names(df_ppp)[end-2:end]

# =========================== Cancel log2 ??????? ==============================
df1_ppp = 2 .^ df_ppp

df1_ppp = mapcols(myNormalize, df1_ppp)

df2_ppp = featureFilterSD(df1_ppp, false, 0.15, 0.15)

names(df2_ppp)[findall(x -> x == "TPM_AT3G13060" || x == "iBAQ_AT3G13060", names(df2_ppp))]
df2_ppp_withECT5 = hcat(df1_ppp[!, "iBAQ_AT3G13060"], df2_ppp)
rename!(df2_ppp_withECT5, :x1 => "iBAQ_AT3G13060")

println("The size of processed table: ", size(df2_ppp))
println("MI is calculating...")

# mi_ppp, samples_ava = calcMI(df2_ppp_withECT5)
# mi_ppp, samples_ava = calcMI(df2_ppp)

# MIsAndSamples = calcMIsLinearly(df2_ppp)


Profile.Allocs.clear()
Profile.Allocs.@profile MIsAndSamples = calcMIsLinearly(df2_ppp[:,1:400])
# Export pprof allocation profile and open interactive profiling web interface.
PProf.Allocs.pprof()
# PProf.refresh(file="alloc-profile.pb_20230713.gz")

@save "result_2.jld2" {compress=true} MIsAndSamples df2_ppp

exit()

# mapreduce(println, hcat, combinations(1:3, 2))



@load "result_2.jld2"



using CairoMakie


posi_iBAQ = findfirst(x -> x == "iBAQ_AT3G13060", names(df2_ppp_withECT5))
vals_iBAQ = mi_ppp[:, posi_iBAQ]
hist(filter(!isnan, vals_iBAQ); bins=128,
    figure = (; resolution = (1200, 800)),
    axis = (; title = "MI Distribution of iBAQ_ECT5", xlabel = "Optimized Mutual information", ylabel = "Frequency"))
#
sortp_iBAQ = sortperm(vals_iBAQ[findall(x -> x >= 1.247, vals_iBAQ)]; rev=true)
top_f10_iBAQ = names(df2_ppp_withECT5)[findall(x -> x >= 1.247, vals_iBAQ)][sortp_iBAQ]


posi_TPM = findfirst(x -> x == "TPM_AT3G13060", names(df2_ppp_withECT5))
vals_TPM = mi_ppp[:, posi_TPM]
hist(filter(!isnan, vals_TPM); bins=128,
    figure = (; resolution = (1200, 800)),
    axis = (; title = "MI Distribution of TPM_ECT5", xlabel = "Optimized Mutual information", ylabel = "Frequency"))
#
sortp_TPM = sortperm(vals_TPM[findall(x -> x >= 1.18, vals_TPM)]; rev=true)
top_f10_TPM = names(df2_ppp_withECT5)[findall(x -> x >= 1.18, vals_TPM)][sortp_TPM]


# Visualize correlation

x0 = df2_ppp_withECT5[!, "iBAQ_AT3G13060"]

x1 = df2_ppp_withECT5[!, "TPM_AT3G52360"]
hexbin(x0, x1;
        bins=75,
        cellsize=0.05,
        colormap = :heat,#[Makie.to_color(:transparent); Makie.to_colormap(:viridis)],
        figure = (; resolution = (1080, 1080)))
#

x2 = df2_ppp_withECT5[!, "iBAQ_AT1G55080"]
hexbin(x0, x2;
        bins=75,
        cellsize=0.05,
        colormap = :heat,
        figure = (; resolution = (1080, 1080)))
#
scatter(x0, x2; color=(:blue, 0.3),
    figure = (; resolution = (1080, 1080)))
#

scatter(df2_ppp_withECT5[!, "TPM_AT3G13060"], df2_ppp_withECT5[!, "TPM_AT5G61230"];
        color = (:blue, 0.3),
        figure = (; resolution = (1080, 1080)))
#
