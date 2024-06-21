__precompile__(true)

# Utils for the network initialization preparation
module PreNetInit

export calcMIsLinearly, myMIXY, tmapreduce, rmOutliersVecXY, rmOutliers


mutable struct myT_LocalOptimMI
    MI_max::Float64
    global_or_local::Bool
    MI_global::Float64
    MI_local::Float64
    local_samples::Vector{Int64}
end

using Random
using DataFrames
# using CSV
using Statistics: quantile, std, mean
# using StatsBase: iqr
# using CausalityTools: mutualinfo, ValueHistogram, RectangularBinning, GaoKannanOhViswanath
using Combinatorics: combinations
# using CUDA
# using LinearAlgebra: UpperTriangular
# using Metrics
# using HypothesisTests: ApproximatePermutationTest#ExactPermutationTest
# using Clustering
# using ComplexityMeasures: RectangularBinning
using Base.Threads: nthreads, @spawn, @threads

function tmapreduce(f, op, itr; tasks_per_thread::Int = 16, kwargs...)
    chunk_size = max(1, length(itr) ÷ (tasks_per_thread * nthreads()))
    tasks = map(Iterators.partition(itr, chunk_size)) do chunk
        @spawn mapreduce(f, op, chunk; kwargs...)
    end
    mapreduce(fetch, op, tasks; kwargs...)
end


## Produce cryptographically secure random numbers (CS(P)RNG)
# function CSPRNG(ns::Int64)
#     return rand(RandomDevice(), ns)
# end


@doc "Normalization"
function myNormalize(v::Union{AbstractArray{Float64}, AbstractArray{Int64}},#CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}, CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}
                    minX::Union{Int64, Float64} = minimum(v), maxX::Union{Int64, Float64} = maximum(v))
    ranX = maxX - minX
    vs = @. (v - minX) / ranX
    # @show maximum(vs)
    return vs
end


@doc """
Sliding windows with changeable size. parameters are in prop, ∈(0,1)
Generate windows and slides
"""
function genWindowsAndSlides(num_sample::Int64,
                             slide_step::Union{Int64, Float64},
                             width_range::Vector{Float64}, width_step::Float64=0.05)
    maximum(width_range) < 1 || throw(ArgumentError("range of window width must .∈ (0,1)"))
    length(width_range) == 2 || throw(ArgumentError("width range must be a range"))
    # 
    if typeof(slide_step) == Int64
        len_step = slide_step
    else
        len_step = ceil(Int64, num_sample * slide_step)
    end
    # 
    width_min, width_max = floor.(Int64, width_range .* num_sample)
    width_stepL = round(Int64, width_step * num_sample)
    if width_stepL < 1
        width_stepL = 1
    end
    num_width = floor(Int64, (width_max - width_min) / width_stepL) + 1
    tmp_w_seq = [i for i = 0:(num_width - 1)]
    widths = zeros(Int64, num_width)
    @. widths = width_min + Int64(width_stepL * tmp_w_seq)
    # Plus 1 for the first window step
    num_move = zeros(Int64, num_width)
    @. num_move = floor(Int64, (num_sample - widths) / len_step) + 1
    # Fill the gap between the end and the last slide
    rems = zeros(Int64, num_width)
    @. rems = (num_sample - widths) % len_step
    f_rem = findall(x -> x > 0, rems)
    num_move[f_rem] .= num_move[f_rem] .+ 1
    sum_move = sum(num_move)
    # Prepare steps for sliding and windows' sizes
    pos_width = zeros(Int64, sum_move)
    pos_move = zeros(Int64, sum_move)
    for xw in eachindex(num_move)
        x_end = sum(num_move[1:xw])
        x_sta = x_end - num_move[xw] + 1
        pos_width[x_sta:x_end] .= widths[xw]
        pos_move[x_sta:x_end] .= [i for i = 0:(num_move[xw] - 1)]
    end
    pos_sta = zeros(Int64, sum_move)
    pos_end = zeros(Int64, sum_move)
    @. pos_end = pos_width + (pos_move * len_step)
    pos_end[pos_end .> num_sample] .= num_sample
    @. pos_sta = pos_end - pos_width + 1
    #
    dfw = DataFrame(window_sta=pos_sta, window_end=pos_end, widths=pos_width, slides=pos_move)
    return dfw
end
### 
function slidingWindow2D(Algor::Function, getOnlyOptimum::Bool,
                        feature1::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                        feature2::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                        slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64,
                        f1_sortperm::Vector{Int64} = sortperm(feature1))
    # length(feature1) == length(feature2) || throw(error("features must have the same length"))
    num_samp = length(feature1)
    # Sort features by the feature1
    f1 = feature1[f1_sortperm]
    f2 = feature2[f1_sortperm]
    # Generate windows for different sizes and slides
    df_win = genWindowsAndSlides(num_samp, slideStep, widthRange, widthStep)
    num_slide = size(df_win, 1)
    # Calc Algor and accept Float64 results
    col_results = zeros(Float64, num_slide)
    @threads for xm in eachindex(col_results)
        col_results[xm] = Algor(f1[(df_win[xm,1]):(df_win[xm,2])], f2[(df_win[xm,1]):(df_win[xm,2])])
    end
    if getOnlyOptimum
        optimumx, posx = findmax(col_results)
        wh_samples = collect(Int64, 1:num_samp)[f1_sortperm][(df_win[posx, 1]):(df_win[posx, 2])]
        sort!(wh_samples)
        return optimumx, wh_samples
    else
        records = hcat(df_win, col_results)
        rename!(records, :x1 => :results)
        return records
    end
end
###
function slidingWindow1D(Algor::Function, getOnlyOptimum::Bool,
                        feature1::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                        slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    # length(feature1) == length(feature2) || throw(error("features must have the same length"))
    num_samp = length(feature1)
    # Sort features by the feature1
    sort!(feature1)
    # Generate windows for different sizes and slides
    df_win = genWindowsAndSlides(num_samp, slideStep, widthRange, widthStep)
    num_slide = size(df_win, 1)
    # Calc Algor
    col_results = zeros(Float64, num_slide)
    @threads for xm in eachindex(col_results)
        col_results[xm] = Algor(feature1[df_win[xm,1]: df_win[xm,2]])
    end
    if getOnlyOptimum
        optimumx, posx = findmax(col_results)
        return optimumx, [df_win[posx, 1], df_win[posx, 2]]# samples in sorted features
    else
        records = hcat(df_win, col_results)
        rename!(records, :x1 => :results)
        return records
    end
end



@doc "Filter features by the given S.D. threshold."

@doc "Norm each col"
function normEachCol!(m::Union{AbstractDataFrame, AbstractMatrix})
    num_f = size(m,2)
    for xf in 1:num_f
        if eltype(view(m, 1, xf)) == Float64
            m[:, xf] = myNormalize(m[:, xf])
            # @show maximum(m[:, xf])
        end
    end
    # @show maximum(m[:, 1])
    # return m
end

@doc "Using DataFrame as input due to different types of columns."
function featureFilterSD(m::Union{AbstractDataFrame, AbstractMatrix}, toNorm::Bool=true,
                         thresholdSD::Float64=0.05, localThrSD::Float64=0.05, localThrSDAbs::Float64=0.01,
                         slideStep::Union{Int64, Float64}=0.05, widthRange::Vector{Float64}=[0.33, 0.67], widthStep::Float64=0.05)
    #
    num_f = size(m, 2)
    sds = zeros(num_f)
    means = zeros(num_f)
    # Normalization. m maybe changed!!!!! Test result: it has not been changed.
    if toNorm
        normEachCol!(m)
    end
    # The First step: global SD
    @threads for xs in eachindex(sds)
        sds[xs] = std(view(m, :, xs))
        means[xs] = mean(view(m, :, xs))
    end
    det1 = @. (means * thresholdSD) < sds
    togets1 = findall(x -> x > 0, det1)
    length(togets1) < 1 && return missing
    # The Second step: local SD
    for xf in eachindex(togets1)
        # To exclude classification data:
        # if typeof(m[1, togets1[xf]]) == Float64
            recs_sd = slidingWindow1D(std, false, view(m, :, togets1[xf]), slideStep, widthRange, widthStep)
            recs_mn = slidingWindow1D(mean, false, view(m, :, togets1[xf]), slideStep, widthRange, widthStep)
            det2 = (recs_mn[:, "results"] .* localThrSD) .> recs_sd[:, "results"]
            det3 = localThrSDAbs .> recs_sd[:, "results"]
            if sum(det2) > 0; togets1[xf] = 0; end;
            if sum(det3) > 0; togets1[xf] = 0; end;
        # end
    end
    togets2 = filter(!iszero, togets1)
    length(togets2) < 1 && return missing
    #
    dfSmall = m[:, togets2]
    return dfSmall
end



@doc "Remove outliers"
# Detect outliers
function detectOutliers(v::Union{AbstractArray{Float64}, AbstractArray{Int64}})
    q1, q3 = quantile(v, [0.25, 0.75])
    # iqr_v = q3 - q1
    iqr15 = 1.5 * (q3 - q1)
    lowest = q1 - iqr15
    highest = q3 + iqr15
    outls = findall(x -> !(lowest < x < highest), v)
    return outls
end
### Main
function rmOutliers(v::AbstractArray)
    detection = detectOutliers(v)
    if length(detection) > 0
        v = v[Not(detection)]
    end
    return v
end
### Main for 2 vectors
function rmOutliersVecXY(X::Union{AbstractVector{Float64}, AbstractVector{Int64}}, Y::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                         detectX = detectOutliers(X),
                         detectY = detectOutliers(Y),
                         getOnlyPosition::Bool=true,
                         maxPropOutl::Float64=0.125)
    #
    lenX = length(X)
    lenY = length(Y)
    lenX == lenY ||
        throw(ArgumentError("number of elements in each array must match"))
    # rm is cancelled if too much outliers exists.
    if length(detectX) > maxPropOutl * lenX
        detectX = Int64[]
    end
    if length(detectY) > maxPropOutl * lenY
        detectY = Int64[]
    end
    elem2rm = unique(vcat(detectX, detectY))
    # sort!(unique!(elem2rm))
    elem2save = collect(Int64, 1:lenX)[InvertedIndex(elem2rm)]
    sort!(elem2save)
    if length(elem2rm) > 0
        if getOnlyPosition
            return elem2save
        else
            oX = X[elem2save]
            oY = Y[elem2save]
            return oX, oY, elem2save
        end
    else
        if getOnlyPosition
            return elem2save
        else
            return X, Y, elem2save
        end
    end
end



@doc """
Estimate mutual information (MI) between x and y using the entropy/probability estimator RectangularBinning (the adaptive partitioning approach).
Freedman-Diaconis' rule (no assumption on the distribution).
"""
function binsFD(v::Union{AbstractVector{Float64},AbstractVector{Float32},AbstractVector{Int64}})::Int64
    vLen = length(v)
    vMin, q1, q3, vMax = quantile(v, [0.0, 0.25, 0.75, 1.0])
    iqr_v = q3 - q1
    bin_width = 2 * iqr_v * vLen ^ (-1/3)
    FD_float = (vMax - vMin) / bin_width
    if isnan(FD_float) || FD_float < 3 || FD_float > vLen
        num_bins = round(Int64, vLen * 0.1)
    else
        num_bins = round(Int64, FD_float)
    end
    if num_bins < 1
        num_bins = 1
    end
    return num_bins
end
## MI quantifies the "amount of information". The minimum is 0. Larger is "better".
# function MIxy(X::Union{AbstractVector{Float64},AbstractVector{Float32},AbstractVector{Int64}},
#               Y::Union{AbstractVector{Float64},AbstractVector{Float32},AbstractVector{Int64}};
#               bins::Vector{Int64} = binsFD.([X,Y]))
#     est = ValueHistogram(RectangularBinning(bins))
#     MI = mutualinfo(est, X, Y)
#     return MI
# end

#
function histogramX1(arrayin::Union{AbstractArray{Int},AbstractArray{Float64},AbstractArray{Float32}},
                     nbins::Int)::Vector{Int}
    minX, maxX = extrema(arrayin)
    bin_width = (maxX - minX) / nbins
    # whInterv = min.(floor.(Int, (arrayin .- minX) ./ bin_width) .+ 1, nbins)
    whInterv = zeros(Int, length(arrayin))
    if bin_width == 0
        # throw(error("+++++++ !!! bin_width is zero +++++++"))
        # ??:
        bin_width = 1
    end
    @. whInterv = min((1 + floor(Int, (arrayin - minX) / bin_width)), nbins)
    return whInterv
end
#=
function aMI(i::Int, nBinsX::Int, nBinsY::Int, pXY::AbstractArray{Float64}, pX::AbstractArray{Float64}, pY::AbstractArray{Float64})
    xi = div(i - 1, nBinsX) + 1
    yi = mod(i - 1, nBinsY) + 1
    oMI = 0.0
    if pXY[xi,yi] > 0.0
        oMI += pXY[xi,yi] * log2(pXY[xi,yi] / (pX[xi] * pY[yi]))
    else
        oMI = 0.0
    end
    return oMI
end
function sMIs(nBinsX::Int64, nBinsY::Int64, pXY::AbstractArray{Float64}, pX::AbstractArray{Float64}, pY::AbstractArray{Float64})
    oMI = 0.0
    for i in 1:(nBinsX * nBinsY)
        xi = div(i-1, nBinsX) + 1
        yi = mod(i-1, nBinsY) + 1
        if pXY[xi,yi] > 0.0
            oMI += pXY[xi,yi] * log2(pXY[xi,yi] / (pX[xi] * pY[yi]))
        end
    end
    return oMI
end
=#
function myMIXY(x::Union{AbstractVector{Int},AbstractVector{Float64},AbstractVector{Float32}},
                y::Union{AbstractVector{Int},AbstractVector{Float64},AbstractVector{Float32}},
                nBinsX::Int=binsFD(x), nBinsY::Int=binsFD(y))
    # inblocks = hcat(histogramX1(x, nBinsX), histogramX1(y, nBinsY))
    x_interv = histogramX1(x, nBinsX)
    y_interv = histogramX1(y, nBinsY)
    # Fill the blocks
    pXY = zeros(Float64, (nBinsX, nBinsY))
    # @simd for i in 1:size(inblocks, 1)
    for i in eachindex(x_interv)
        # pXY[inblocks[i,1], inblocks[i,2]] += 1.0
        pXY[x_interv[i], y_interv[i]] += 1.0
    end
    # How to normalize pXY ?
    pXY = pXY ./ length(x)
    pX = sum(pXY, dims=2)
    pY = sum(pXY, dims=1)
    #
    oMI = 0.0
    for i in 1:(nBinsX * nBinsY)
        xi = div(i-1, nBinsY) + 1
        yi = mod(i-1, nBinsY) + 1
        if pXY[xi,yi] > 0.0
            oMI += pXY[xi,yi] * log2(pXY[xi,yi] / (pX[xi] * pY[yi]))
        end
    end
    #
    # aaMI(i::Int) = aMI(i, nBinsX, nBinsY, pXY, pX, pY)
    # oMI = tmapreduce(aaMI, +, 1:(nBinsX * nBinsY))
    #
    # oMI = sMIs(nBinsX, nBinsY, pXY, pX, pY)
    return oMI
end


@doc "Optimal MI may appear in local samples"
function optimLocalMI(X::Union{AbstractVector{Float64},AbstractVector{Int64}}, Y::Union{AbstractVector{Float64},AbstractVector{Int64}},
                      getOnlyOptimum::Bool,
                      slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64, f::Function=myMIXY)
    #
    MI_full = f(X, Y)
    #
    sortpermX = sortperm(X)
    optimum, wh2save = slidingWindow2D(f, true, X, Y, slideStep, widthRange, widthStep, sortpermX)
    max_MI = maximum([optimum, MI_full])
    if getOnlyOptimum
        return max_MI
    else
        global_or_local = optimum < MI_full
        outs = myT_LocalOptimMI(max_MI, global_or_local, MI_full, optimum, wh2save)
        return outs
    end
end



@doc "Calc mutual information dependence"
### matrix modifier
function atomMI(f1_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}}, f2_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                outl_f1::Vector{Int64}, outl_f2::Vector{Int64},
                slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    #
    xro, xco, wh2save = rmOutliersVecXY(f1_vec, f2_vec, outl_f1, outl_f2, false)
    # ============= !!!!! Re-normalization after outliers removed !!!!! =============
    xro .= myNormalize(xro)
    xco .= myNormalize(xco)
    #
    optimumxs = optimLocalMI(xro, xco, false, slideStep, widthRange, widthStep)
    if !(optimumxs.global_or_local)
        # wh2save = intersect(wh2save, optimumxs.local_samples)
        wh2save = wh2save[optimumxs.local_samples]
        xro = xro[optimumxs.local_samples]
        xco = xco[optimumxs.local_samples]
    end
    if length(f1_vec) == length(wh2save)
        wh2save = Int64[]# => all samples are saved
    end
    # return reshape([optimumxs.MI_max, wh2save, xro, xco], (1,4))
    # return optimumxs.MI_max, wh2save
    outs = DataFrame(MImax=optimumxs.MI_max, SamplesUsed=[wh2save], feature1=[xro], feature2=[xco])
    return outs
end
function atomMI(f1_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}}, f2_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                slideStep::Union{Int64, Float64}=0.02, widthRange::Vector{Float64}=[0.6,0.99], widthStep::Float64=0.02)
    #
    optimumxs = optimLocalMI(f1_vec, f2_vec, false, slideStep, widthRange, widthStep)
    if optimumxs.global_or_local
        wh2save = Int64[]
        xro = f1_vec
        xco = f2_vec
    else
        wh2save = optimumxs.local_samples
        xro = f1_vec[wh2save]
        xco = f2_vec[wh2save]
    end
    # return reshape([optimumxs.MI_max, wh2save], (1,2))
    # return optimumxs.MI_max, wh2save
    outs = DataFrame(MImax=optimumxs.MI_max, SamplesUsed=[wh2save], feature1=[xro], feature2=[xco])
    return outs
end

# function hcatcollects()

### X-X
function atomsMIs(X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  outlierXs::Vector{Vector{Int64}},
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    tAtom(aCombn::Vector{Int64}) = atomMI(view(X,:,aCombn[1]), view(X,:,aCombn[2]), outlierXs[aCombn[1]], outlierXs[aCombn[2]], slideStep, widthRange, widthStep)
    resultMIandSample = tmapreduce(tAtom, vcat, combinations(1:size(X,2), 2))
    return resultMIandSample
end
function atomsMIs(X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    tAtom(aCombn::Vector{Int64}) = atomMI(view(X,:,aCombn[1]), view(X,:,aCombn[2]), slideStep, widthRange, widthStep)
    resultMIandSample = tmapreduce(tAtom, vcat, combinations(1:size(X,2), 2))
    return resultMIandSample
end
### X-Y
function atomsMIs(X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  Y::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  outlierXs::Vector{Vector{Int64}},
                  outlierYs::Vector{Vector{Int64}},
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    #
    combns = [[x,y] for x in 1:size(X, 2) for y in 1:size(Y, 2)]
    tAtom(aCombn::Vector{Int64}) = atomMI(view(X,:,aCombn[1]), view(Y,:,aCombn[2]), outlierXs[aCombn[1]], outlierYs[aCombn[2]], slideStep, widthRange, widthStep)
    resultMIandSample = tmapreduce(tAtom, vcat, combns)
    return resultMIandSample
end
function atomsMIs(X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  Y::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    #
    combns = [[x,y] for x in 1:size(X, 2) for y in 1:size(Y, 2)]
    tAtom(aCombn::Vector{Int64}) = atomMI(view(X,:,aCombn[1]), view(Y,:,aCombn[2]), slideStep, widthRange, widthStep)
    resultMIandSample = tmapreduce(tAtom, vcat, combns)
    return resultMIandSample
end


@doc "MI for X-X. Linear format makes it faster."
function calcMIsLinearly(X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                        slideStep::Union{Int64, Float64}=0.02, widthRange::Vector{Float64}=[0.6,0.99], widthStep::Float64=0.02,
                        rm_outliers::Bool=true)
    nr, nc = size(X)
    # Calc MI
    if rm_outliers
        # pre outlier detection for acceleration
        outlierXs = repeat([zeros(Int64, nr)], nc)
        @threads for xc in eachindex(outlierXs)
            outlierXs[xc] = detectOutliers(view(X, :, xc))
        end
        MIandSample = atomsMIs(X, outlierXs, slideStep, widthRange, widthStep)
    else
        MIandSample = atomsMIs(X, slideStep, widthRange, widthStep)
    end
    return MIandSample
end

#=
### Get a feature from the upper triangular
function getFeatureInUpperTri(m::AbstractMatrix, xfeat::Int64, skipCheck::Bool=true)
    nr, nc = size(m)
    if !skipCheck
        nr == nc || throw(error("not square"))
        xfeat > 0 || throw(error("incorrect xfeat" * ": " * string(xfeat)))
        xfeat ≤ nc || throw(error("incorrect xfeat" * ": " * string(xfeat)))
    end
    featvec = Vector{eltype(m)}(undef, nc)
    if 1 < xfeat < nc
        featvec[1:(xfeat - 1)] .= view(m, 1:(xfeat - 1), xfeat)
        featvec[(1 + xfeat):end] .= view(m, xfeat, (xfeat + 1):nc)
        featvec[xfeat] = NaN
    elseif xfeat == 1
        featvec[2:end] .= view(m, 1, 2:nc)
        featvec[1] = NaN
    elseif xfeat == nc
        featvec[1:end-1] .= view(m, 1:(nc-1), nc)
        featvec[nc] = NaN
    end
    return featvec
end


### Calc MI for X-X
function calcMI(X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                slideStep::Union{Int64, Float64}=0.02, widthRange::Vector{Float64}=[0.6,0.98], widthStep::Float64=0.03,
                rm_outliers::Bool=true, to_norm::Bool=false)
    m_tri, samples = calcMITriangle(X, slideStep, widthRange, widthStep, rm_outliers)
    nc = size(m_tri, 2)
    m_o = Matrix{Float64}(undef, nc, nc)
    if to_norm
        MI_self = Vector{Float64}(undef, nc)
        # Calc MI for xi-xi
        @threads for xc = 1:nc
            MI_self[xc] = MIxy(view(X, :, xc), view(X, :, xc))
        end
        # Get values min-max normalized
        @threads for xc = 1:nc
            m_o[:, xc] .= myNormalize(getFeatureInUpperTri(m_tri, xc), 0.0, MI_self[xc])
        end
    else
        @threads for xc = 1:nc
            m_o[:, xc] .= getFeatureInUpperTri(m_tri, xc)
        end
    end
    return m_o, samples
end

###  Calc MI for X-Y
function calcMI(X::Union{AbstractMatrix{Float64}, AbstractDataFrame}, Y::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                slideStep::Union{Int64, Float64}=0.02, widthRange::Vector{Float64}=[0.6,0.98], widthStep::Float64=0.03,
                rm_outliers::Bool=true)
    nr, nc = size(X,2), size(Y,2)
    n_sample = size(X,1)
    C = zeros(Float64, nr, nc)
    sampl_get = repeat([zeros(Int64, nr)], nr * nc)
    # combn
    combn_vec = repeat([zeros(Int64, 2)], nr * nc)
    @threads for xc = 1:nc
        for xr = 1:nr
            combn_vec[(xc-1)*nr + xr] = [xr, xc]
        end
    end
    # Calc MI
    if rm_outliers
        # pre outlier detection for acceleration
        outlierXs = repeat([zeros(Int64, n_sample)], nr)
        outlierYs = repeat([zeros(Int64, n_sample)], nc)
        @threads for xc in eachindex(outlierXs)
            outlierXs[xc] = detectOutliers(view(X, :, xc))
        end
        @threads for yc in eachindex(outlierYs)
            outlierYs[yc] = detectOutliers(view(Y, :, yc))
        end
        atomsMI!(C, sampl_get, combn_vec, X, Y, outlierXs, outlierYs, slideStep, widthRange, widthStep)
    else
        atomsMI!(C, sampl_get, combn_vec, X, Y, slideStep, widthRange, widthStep)
    end
    sample_save = DataFrame(combn=combn_vec, samples=sampl_get)
    return C, sample_save
end


### Sum for sorted vec
function sumSortedVec(v::AbstractVector, ascending::Bool=false)
    if ascending
        vst = sort(v)
    else
        vst = sort(v, rev=true)
    end
    vsum = Vector{eltype(vst)}(undef, length(vst))
    @threads for i = 1:length(vst)
        vsum[i] = sum(vst[1:i])
    end
    return vsum
end
=#


###
# function gradSortedVec(v::AbstractVector, ascending::Bool=false)
    
# end


## Approximate Permutation Test



end
