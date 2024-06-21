# Test & Debug

path_mrna = "/home/wuch/Downloads/FT/mrna_1000.csv"
path_meth = "/home/wuch/Downloads/FT/meth_3000.csv"

include("../src/act_env.jl")
include("../src/md_1_PreNetInit.jl")
using .PreNetInit

# using CairoMakie
using JLD2, CSV, DataFrames

# using Profile
# using PProf


# Read data
mrna = CSV.read(path_mrna, DataFrame; drop=[1])
meth = CSV.read(path_meth, DataFrame; drop=[1])

# omics_mat = hcat(mrna, meth)
omics_mat = hcat(Matrix(mrna), Matrix(meth))


println("Start testing 127x127 matrix.")
@time test1_values_and_used_samples = calcMIsLinearly(omics_mat[:, 1:127])
println("  Done.")

GC.gc()

println("Start testing another 127x127 matrix.")
@time test1_values_and_used_samples = calcMIsLinearly(omics_mat[:, 998:1124])
println("  Done.")


# println("Start testing 1027x1027 matrix.")
# @time test2_values_and_used_samples = calcMIsLinearly(omics_mat[:, 1:1027])
# println("  Done.")


# # Collect an allocation profile
# Profile.Allocs.clear()
# Profile.Allocs.@profile test2_values_and_used_samples = calcMIsLinearly(omics_mat[:, 998:1124])
# # Export pprof allocation profile and open interactive profiling web interface.
# PProf.Allocs.pprof()


# println("Start.")
# values_and_used_samples = calcMIsLinearly(omics_mat)

# println("  Done. Saving results in JLD2 format...")
# @save "MI_test.jld2" {compress=true} values_and_used_samples


# @load "MI_omics.jld2"
