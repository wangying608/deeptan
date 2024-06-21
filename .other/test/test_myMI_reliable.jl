include("../src/md_1_PreNetInit.jl")

x0 = [i for i in 1:1000] .+ 0.1
y0 = x0
myMIXY(x0, y0)

x1 = x0
y1 = x1 .+ 1.1
# binsFD(x1)
myMIXY(x1, y1)

x2 = x0
y2 = x2 .+ 200 .* rand(Float64, 1000)
myMIXY(x2, y2)

x3 = x0
y3 = x3 .- 900 .* rand(Float64, 1000)
myMIXY(x3, y3)
