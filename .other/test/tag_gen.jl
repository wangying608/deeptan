include("../src/tools.jl")

# Generate 24 tags for datasets
tags24 = getRandomName(24, "/home/wuch/Documents/")

using CSV
using DataFrames
CSV.write("/home/wuch/Documents/rand_tags.csv", DataFrame(tags=tags24))

tags24 = CSV.read("/home/wuch/Documents/rand_tags.csv", DataFrame; header=false)[:,1] |> Vector{String}
tagsnew8 = getRandomName(8, "/home/wuch/Documents", true, true, tags24)
tags32 = vcat(tags24, tagsnew8)
CSV.write("/home/wuch/Documents/rand_tags32.csv", DataFrame(tags=tags32))
