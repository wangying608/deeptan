__precompile__(true)

# Utils for the network initialization preparation
# module PreNetInit

# export CSPRNG, myNormalize
# export featureFilterSD, sumSortedVec
# export calcMI

include("types.jl")
using Random
using DataFrames
using Statistics: quantile, std, mean
using StatsBase: iqr
using CausalityTools: mutualinfo, ValueHistogram, RectangularBinning, GaoKannanOhViswanath
using Combinatorics: combinations
# using LinearAlgebra: UpperTriangular
# using Metrics
# using HypothesisTests: ApproximatePermutationTest#ExactPermutationTest
# using Clustering
# using ComplexityMeasures: RectangularBinning
using Base.Threads: @threads


## Produce cryptographically secure random numbers (CS(P)RNG)
function CSPRNG(ns::Int64)
    return rand(RandomDevice(), ns)
end


## Normalization
function myNormalize(v::Union{AbstractArray{Float64}, AbstractArray{Int64}},
                    minX::Union{Int64, Float64} = minimum(v), maxX::Union{Int64, Float64} = maximum(v))
    ranX = maxX - minX
    vs = @. (v - minX) / ranX
    return vs
end


## Sliding windows with changeable size. parameters are in prop, ∈(0,1)
### Generate windows and slides
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
    # for xm in eachindex(pos_sta)
    #     pos_end[xm] = pos_width[xm] + pos_move[xm] * len_step
    #     if pos_end[xm] > num_sample
    #         pos_end[xm] = num_sample
    #     end
    #     pos_sta[xm] = pos_end[xm] - pos_width[xm] + 1
    # end
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
    # Calc Algor
    col_results = zeros(Float64, num_slide)
    @threads for xm in eachindex(col_results)
        col_results[xm] = Algor(f1[(df_win[xm,1]):(df_win[xm,2])], f2[(df_win[xm,1]):(df_win[xm,2])])
    end
    if getOnlyOptimum
        optimumx, posx = findmax(col_results)
        wh_samples = sort([x for x in 1:num_samp][f1_sortperm][(df_win[posx, 1]):(df_win[posx, 2])])
        return optimumx, wh_samples# samples in sorted features
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



## Filter features by the given S.D. threshold.

### Norm each col
function normEachCol!(m::Union{AbstractDataFrame, AbstractMatrix})
    num_f = size(m,2)
    for xf in 1:num_f
        if typeof(view(m, 1, xf)) == Float64
            m[:, xf] .= myNormalize(view(m, :, xf))
        end
    end
end

### Using DataFrame as input due to different types of columns.
function featureFilterSD(m::Union{AbstractDataFrame, AbstractMatrix}, toNorm::Bool=true,
                         thresholdSD::Float64=0.05, localThrSD::Float64=0.05, localThrSDAbs::Float64=0.01,
                         slideStep::Union{Int64, Float64}=0.05, widthRange::Vector{Float64}=[0.33, 0.67], widthStep::Float64=0.05)
    #
    num_f = size(m, 2)
    sds = zeros(num_f)
    means = zeros(num_f)
    # Normalization. m maybe changed!!!!! Test result: it has not changed.
    if toNorm
        normEachCol!(m)
    end
    # The First step: global SD
    @threads for xs in eachindex(sds)
        sds[xs] = std(m[:, xs])
        means[xs] = mean(m[:, xs])
    end
    det1 = @. (means * thresholdSD) < sds
    togets1 = findall(x -> x > 0, det1)
    length(togets1) < 1 && return missing
    # The Second step: local SD
    for xf in eachindex(togets1)
        # To exclude classification data:
        # if typeof(m[1, togets1[xf]]) == Float64
            recs_sd = slidingWindow1D(std, false, m[:, togets1[xf]], slideStep, widthRange, widthStep)
            recs_mn = slidingWindow1D(mean, false, m[:, togets1[xf]], slideStep, widthRange, widthStep)
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



## Remove outliers
### Detect outliers
function detectOutliers(v::Union{AbstractArray{Float64}, AbstractArray{Int64}})
    q1, q3 = quantile(v, [0.25, 0.75])
    #iqr(v)
    iqr_v = q3 - q1
    lowest = q1 - 1.5 * iqr_v
    highest = q3 + 1.5 * iqr_v
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
    size(X, 1) == size(Y, 1) ||
        throw(ArgumentError("number of rows in each array must match"))
    elem2rm = Int64[]
    # rm is cancelled if too much outliers exists.
    if length(detectX) > maxPropOutl * length(X)
        detectX = Int64[]
    end
    if length(detectY) > maxPropOutl * length(Y)
        detectY = Int64[]
    end
    append!(elem2rm, detectX)
    append!(elem2rm, detectY)
    sort!(unique!(elem2rm))
    elem2save = [i for i in 1:length(X)][Not(elem2rm)]
    if length(elem2rm) > 0
        if getOnlyPosition
            return elem2save
        else
            return X[elem2save], Y[elem2save], elem2save
        end
    else
        if getOnlyPosition
            return elem2save
        else
            return X, Y, elem2save
        end
    end
end



## Estimate mutual information (MI) between x and y using the entropy/probability estimator RectangularBinning (the adaptive partitioning approach).
### Freedman-Diaconis' rule (no assumption on the distribution).
function binsFD(v::Union{AbstractVector{Float64},AbstractVector{Int64}})::Int64
    bin_width = 2 * iqr(v) * length(v)^(-1/3)
    FD_float = (maximum(v) - minimum(v)) / bin_width
    if isnan(FD_float) || FD_float < 3 || FD_float > length(v)
        num_bins = round(Int64, length(v) * 0.1)
    else
        num_bins = round(Int64, FD_float)
    end
    return num_bins
end
### MI quantifies the "amount of information". The minimum is 0. Larger is "better".
function MIxy(X::Union{AbstractVector{Float64},AbstractVector{Int64}}, Y::Union{AbstractVector{Float64},AbstractVector{Int64}};
              bins::Vector{Int64} = binsFD.([X,Y]))
    est = ValueHistogram(RectangularBinning(bins))
    MI = mutualinfo(est, X, Y)
    return MI
end


### Optimize MI may appear in local samples
####
function optimLocalMI(X::Union{AbstractVector{Float64},AbstractVector{Int64}}, Y::Union{AbstractVector{Float64},AbstractVector{Int64}},
                      getOnlyOptimum::Bool,
                      slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    #
    MI_full = MIxy(X, Y)
    #
    sortpermX = sortperm(X)
    optimum, whSamp = slidingWindow2D(MIxy, true, X, Y, slideStep, widthRange, widthStep, sortpermX)
    max_MI = maximum([optimum, MI_full])
    if getOnlyOptimum
        return max_MI
    else
        global_or_local = optimum < MI_full
        out = myT_LocalOptimMI(max_MI, global_or_local, MI_full, optimum, whSamp)
        return out
    end
end



## Calc mutual dependence

### matrix modifier
function atomMI(f1_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}}, f2_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                outl_f1::Vector{Int64}, outl_f2::Vector{Int64},
                slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    #
    xro, xco, wh2save = rmOutliersVecXY(f1_vec, f2_vec, outl_f1, outl_f2, false)
    optimumxs = optimLocalMI(xro, xco, false, slideStep, widthRange, widthStep)
    if !(optimumxs.global_or_local)
        wh2save = intersect(wh2save, optimumxs.local_samples)
    end
    if length(f1_vec) == length(wh2save)
        wh2save = Int64[]# Means all samples are saved
    end
    return optimumxs.MI_max, wh2save
end
function atomMI(f1_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}}, f2_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    #
    optimumxs = optimLocalMI(f1_vec, f2_vec, false, slideStep, widthRange, widthStep)
    if optimumxs.global_or_local
        wh2save = Int64[]
    else
        wh2save = optimumxs.local_samples
    end
    return optimumxs.MI_max, wh2save
end

### X-X
function atomsMI!(C::Matrix{Float64}, sampl_get::Vector{Vector{Int64}}, combns::Vector{Vector{Int64}},
                  X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  outlierXs::Vector{Vector{Int64}}, # rm_outliers = true
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    @threads for cx in eachindex(combns)
        C[combns[cx][1], combns[cx][2]], sampl_get[cx] = atomMI(view(X, :, combns[cx][1]), view(X, :, combns[cx][2]), outlierXs[combns[cx][1]], outlierXs[combns[cx][2]], slideStep, widthRange, widthStep)
    end
end
function atomsMI!(C::Matrix{Float64}, sampl_get::Vector{Vector{Int64}}, combns::Vector{Vector{Int64}},
                  X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    @threads for cx in eachindex(combns)
        C[combns[cx][1], combns[cx][2]], sampl_get[cx] = atomMI(view(X, :, combns[cx][1]), view(X, :, combns[cx][2]), slideStep, widthRange, widthStep)
    end
end
### X-Y
function atomsMI!(C::Matrix{Float64}, sampl_get::Vector{Vector{Int64}}, combns::Vector{Vector{Int64}},
                  X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  Y::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  outlierXs::Vector{Vector{Int64}}, # rm_outliers = true
                  outlierYs::Vector{Vector{Int64}},
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    @threads for cx in eachindex(combns)
        C[combns[cx][1], combns[cx][2]], sampl_get[cx] = atomMI(view(X, :, combns[cx][1]), view(Y, :, combns[cx][2]), outlierXs[combns[cx][1]], outlierYs[combns[cx][2]], slideStep, widthRange, widthStep)
    end
end
function atomsMI!(C::Matrix{Float64}, sampl_get::Vector{Vector{Int64}}, combns::Vector{Vector{Int64}},
                  X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  Y::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                  slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64)
    @threads for cx in eachindex(combns)
        C[combns[cx][1], combns[cx][2]], sampl_get[cx] = atomMI(view(X, :, combns[cx][1]), view(Y, :, combns[cx][2]), slideStep, widthRange, widthStep)
    end
end


### MI for X-X. UpperTriangular makes it faster.
function calcMITriangle(X::Union{AbstractMatrix{Float64}, AbstractDataFrame},
                        slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64,
                        rm_outliers::Bool=true, nan_or_zero::Bool=true)
    nr, nc = size(X)
    C = zeros(Float64, nc, nc)
    # combn
    combn_vec = collect(combinations(1:nc, 2))
    sampl_get = repeat([zeros(Int64, nr)], length(combn_vec))
    # Calc MI
    if rm_outliers
        # pre outlier detection for acceleration
        outlierXs = repeat([zeros(Int64, nr)], nc)
        @threads for xc in eachindex(outlierXs)
            outlierXs[xc] = detectOutliers(view(X, :, xc))
        end
        atomsMI!(C, sampl_get, combn_vec, X, outlierXs, slideStep, widthRange, widthStep)
    else
        atomsMI!(C, sampl_get, combn_vec, X, slideStep, widthRange, widthStep)
    end
    if nan_or_zero
        @threads for xr = 1:nc
            C[xr, 1:xr] .= NaN
        end
    end
    sample_save = DataFrame(combn=combn_vec, samples=sampl_get)
    return C, sample_save
end

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
    m_tri, samples = calcMITriangle(X, slideStep, widthRange, widthStep, rm_outliers, true)
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


###
# function gradSortedVec(v::AbstractVector, ascending::Bool=false)
    
# end


## Approximate Permutation Test


# end
