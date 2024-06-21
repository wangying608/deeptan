# Trashed utils

## Simple regression
### Regression for the given formula
function regres_lmXY(X::AbstractVector, Y::AbstractVector, formulaDef::FormulaTerm=@formula(y ~ 1 + x), onlyGetLoss::Bool=false, lossF::Function=mse)
    df = DataFrame(x=X, y=Y)
    lr = lm(formulaDef, df)
    yhatm = predict(lr, df)
    yhat = Vector{Float64}(undef, length(yhatm))
    yhat .= yhatm
    loss = lossF(yhat, Y)
    if onlyGetLoss
        return loss
    else
        return loss, lr
    end
end
### Auto Selection
function auto_regres(X::AbstractVector, Y::AbstractVector, onlyGetLoss::Bool=false, lossF::Function=mse)
    fm1 = regres_lmXY(X, Y, @formula(y ~ 1 + x), onlyGetLoss, lossF)
    fm2 = regres_lmXY(X, Y, @formula(y ~ 1 + x + x^2), onlyGetLoss, lossF)
    fm3 = regres_lmXY(X, Y, @formula(y ~ 1 + x + x^2 + x^3), onlyGetLoss, lossF)
    fmAll = fm1, fm2, fm3
    if onlyGetLoss
        return minimum(fmAll)
    else
        losses = [fm1[1], fm2[1], fm3[1]]
        lossLrm = fmAll[findmin(losses)[2]]
        return lossLrm
    end
end

### Loss matrix modifier
function atomXY!(mat::Matrix{Float64}, xr::Int64, xc::Int64, X::AbstractArray, Y::AbstractArray, remove_outliers::Bool=true)
    Xj = view(X, :, xr)
    if any(isnan, Xj)
        mat[xr,:] .= NaN
    end
    Yi = view(Y, :, xc)
    if any(isnan, Yi)
        mat[xr,xc] = NaN
    else
        if remove_outliers
            XjS, YiS = rm_outliers_2vec(Xj, Yi)
            mat[xr,xc] = auto_regres(XjS, YiS, true)
        else
            mat[xr,xc] = auto_regres(Xj, Yi, true)
        end
    end
end

### Calc each CC
function cor_autox123(X::AbstractArray, Y::AbstractArray, remove_outliers::Bool=true)
    size(X, 1) == size(Y, 1) ||
        throw(ArgumentError("number of rows in each array must match"))
    nr = size(X, 2)
    nc = size(Y, 2)
    C = Matrix{Float64}(undef, nr, nc)
    Threads.@threads for j = 1:nr
        for i = 1:nc
            atomXY!(C, j, i, X, Y, remove_outliers)
        end
    end
    return C
end





function detect_outliers(v::AbstractArray, distMADs::Float64=3.0)
    med = median(v)
    MAD = mad(v)
    outls = findall(x -> !(med - distMADs * MAD < x < distMADs * MAD + med), v)
    return outls
end

### Main for mat
function rm_outliers_mat(mat::AbstractArray, distMADs::Float64=3.5)
    lines2rm = Int64[]
    nc = size(mat, 2)
    for i = 1:nc
        detects = detect_outliers(view(mat, :, i), distMADs)
        if length(detects) > 0
            append!(lines2rm, detects)
        end
    end
    sort!(unique!(lines2rm))
    return mat[Not(lines2rm), :]
end

## Calc correlation based on predictability(PCC?) of the cubic function
function cor_autox123(X::AbstractArray, Y::AbstractArray, remove_outliers::Bool=true, distMADs::Float64=3.0)
    size(X, 1) == size(Y, 1) ||
        throw(ArgumentError("number of rows in each array must match"))
    nr = size(X, 2)
    nc = size(Y, 2)
    C = Matrix{Float64}(undef, nr, nc)
    Threads.@threads for j = 1:nr
        Xj = view(X, :, j)
        if any(isnan, Xj)
            C[j,:] .= NaN
        else
            for i = 1:nc
                Yi = view(Y, :, i)
                if any(isnan, Yi)
                    C[j,i] = NaN
                else
                    if remove_outliers
                        Xj, Yi = rm_outliers_2vec(Xj, Yi, distMADs)
                    end
                    C[j,i] = auto_regres(Xj, Yi, true)
                end
            end
        end
    end
    return C
end

### Cubic equation
function regres_cubic(X::AbstractVector, Y::AbstractVector, onlyGetLoss::Bool=false, lossF::Function=mse)
    df = DataFrame(x=X, y=Y)
    lr = lm(@formula(y ~ 1 + x + x^2 + x^3), df)
    yhatm = predict(lr, df)
    yhat = Vector{Float64}(undef, length(yhatm))
    yhat .= yhatm
    loss = lossF(yhat, Y)
    if onlyGetLoss
        return loss
    else
        return loss, lr
    end
end
### Quadratic
function regres_quadratic(X::AbstractVector, Y::AbstractVector, onlyGetLoss::Bool=false, lossF::Function=mse)
    df = DataFrame(x=X, y=Y)
    lr = lm(@formula(y ~ 1 + x + x^2), df)
    yhatm = predict(lr, df)
    yhat = Vector{Float64}(undef, length(yhatm))
    yhat .= yhatm
    loss = lossF(yhat, Y)
    if onlyGetLoss
        return loss
    else
        return loss, lr
    end
end
### Linear
function regres_linear(X::AbstractVector, Y::AbstractVector, onlyGetLoss::Bool=false, lossF::Function=mse)
    df = DataFrame(x=X, y=Y)
    lr = lm(@formula(y ~ 1 + x), df)
    yhatm = predict(lr, df)
    yhat = Vector{Float64}(undef, length(yhatm))
    yhat .= yhatm
    loss = lossF(yhat, Y)
    if onlyGetLoss
        return loss
    else
        return loss, lr
    end
end



# 20230502

mapcols!(zscore, mrna)

function normalize_minX(m::Matrix, dim::Int=1, minX::Float64=0.0)
    dim == 1 || dim == 2 || throw(error("wrong dim"))
    nr, nc = size(m)
    omat = Matrix{typeof(m[1,1])}(undef, nr, nc)
    if dim == 1
        for xr = 1:nr
            omat[xr, :] .= normalize(m[xr, :], minX)
        end
    else
        for xc = 1:nc
            omat[:, xc] .= normalize(m[:, xc], minX)
        end
    end
    return omat
end

### Calc MIs
function MI_mat(X::AbstractArray, Y::AbstractArray=X, remove_outliers::Bool=true, norm_MI::Bool=true, upper_triangle::Bool=true)
    size(X, 1) == size(Y, 1) ||
        throw(ArgumentError("number of rows in each array must match"))
    nr = size(X, 2)
    nc = size(Y, 2)
    C = Matrix{Float64}(undef, nr, nc)
    Threads.@threads for j = 1:nr
        for i = 1:nc
            atomXY!(C, j, i, X, Y, remove_outliers)
        end
    end
    if norm_MI
        C = normalize_minX(C)
    end
    if upper_triangle
        C_vec = upper_tri(C, true)
        C_mat = upper_tri(C, false)
        return C_mat, C_vec
    else
        return C
    end
    return nothing
end



# 20230503

## UpperTriangular excludes x[i]-x[i]
function upper_tri(m::Matrix, outputVec::Bool)
    nr, nc = size(m)
    nr == nc || throw(error("not square"))
    if outputVec
        len_o = (nr * nr - nr) / 2 |> Int64
        oVec = Vector{typeof(m[1,1])}(undef, len_o)
        ncStart = 2
        nP0 = 1
        nP1 = nP0 + nr - ncStart
        for xr = 1:(nr-1)
            oVec[nP0:nP1] .= m[xr, ncStart:end]
            ncStart = ncStart + 1
            nP0 = nP1 + 1
            nP1 = nP0 + nr - ncStart
        end
        return oVec
    else
        oTri = Matrix{typeof(m[1,1])}(undef, nr, nc)
        for xr = 1:nr
            oTri[xr, 1:xr] .= NaN
            oTri[xr, (xr+1):end] .= m[xr, (xr+1):end]
        end
        return oTri
    end
end

function atomMI!(mat::Union{AbstractMatrix, AbstractDataFrame}, xr::Int64, xc::Int64, X::Union{AbstractMatrix, AbstractDataFrame}, Y::Union{AbstractMatrix, AbstractDataFrame}, bins_xy::Vector{Int64}, remove_outliers::Bool=true)
    Xj = view(X, :, xr)
    if any(isnan, Xj)
        mat[xr,:] .= NaN
    end
    Yi = view(Y, :, xc)
    if any(isnan, Yi)
        mat[xr,xc] = NaN
    else
        if remove_outliers
            XjS, YiS = rmOutliersVecXY(Xj, Yi)
            mat[xr,xc] = MIxy(XjS, YiS, bins=bins_xy)
        else
            mat[xr,xc] = MIxy(Xj, Yi, bins=bins_xy)
        end
    end
end


# 20230508


### MI for X-X. A UpperTriangular is returned.
function calcMI(X::Union{AbstractMatrix{Float64}, AbstractDataFrame}, remove_outliers::Bool=true)
    nc = size(X, 2)#col: features
    # Bin numbers of x
    bins_x = Vector{Int64}(undef, nc)
    Threads.@threads for xc = 1:nc
        bins_x[xc] = binsFD(view(X, :, xc))
    end
    # if typeof(X) <: AbstractDataFrame
    #     C = DataFrame(Matrix{Float64}(undef, nc, nc), names(X))
    # else
    #     C = Matrix{Float64}(undef, nc, nc)
    # end
    C = Matrix{Float64}(undef, nc, nc)
    # combn
    combn_vec = collect(combinations(1:nc, 2))
    # Calc MI
    Threads.@threads for cx in eachindex(combn_vec)
        atomMI!(C, combn_vec[cx][1], combn_vec[cx][2], view(X, :, combn_vec[cx][1]), view(X, :, combn_vec[cx][2]), [bins_x[combn_vec[cx][1]], bins_x[combn_vec[cx][2]]], remove_outliers)
    end
    #
    Threads.@threads for xr = 1:nc
        C[xr, 1:xr] .= NaN
    end
    #
    return C
end


# 20230511
### MI for X-X. Normalization for each feature. UpperTriangular makes it faster.
function calcMITriangle(X::Union{AbstractMatrix{Float64}, AbstractDataFrame}, remove_outliers::Bool=true, nan_or_zero::Bool=true)
    nc = size(X, 2)#col: features
    # Bin numbers of x
    bins_x = zeros(Int64, nc)
    Threads.@threads for xc = 1:nc
        bins_x[xc] = binsFD(view(X, :, xc))
    end
    C = zeros(Float64, nc, nc)
    # combn
    combn_vec = collect(combinations(1:nc, 2))
    # Calc MI
    Threads.@threads for cx in eachindex(combn_vec)
        atomMI!(C, combn_vec[cx][1], combn_vec[cx][2], view(X, :, combn_vec[cx][1]), view(X, :, combn_vec[cx][2]), [bins_x[combn_vec[cx][1]], bins_x[combn_vec[cx][2]]], remove_outliers)
    end
    #
    if nan_or_zero
        Threads.@threads for xr = 1:nc
            C[xr, 1:xr] .= NaN
        end
    end
    #
    return C
end


# 20230512

### (OPTIONAL) Optimize MI may appear in local samples
####
function localMIPinnedPropSample(xsorted::Union{AbstractVector{Float64},AbstractVector{Int64}}, ysorted::Union{AbstractVector{Float64},AbstractVector{Int64}},
                                prop_sample::Float64=0.6, step_window::Float64=5/length(xsorted))
    #
    xlen = length(xsorted)
    n_step = round(Int64, ((1 - prop_sample) / step_window)) + 1# "round" can results errors
    # Windows for n steps
    len_win = round(Int64, prop_sample * xlen)
    windows = zeros(Int64, n_step, 2)
    len_step = round(Int64, xlen * step_window)
    len_step < len_win || throw(ArgumentError("prop_sample is too small"))
    Threads.@threads for xt = 1:n_step
        windows[xt, :] .= [(1 + (xt - 1) * len_step), (len_win + (xt - 1) * len_step)]
    end
    windows[n_step, 2] = xlen
    MIs = zeros(Float64, n_step)
    Threads.@threads for xt = 1:n_step
        # windows[xt,2] <= xlen ||
            # throw(error("error MARKER!  xt=" * string(xt) * "  lenX=" * string(xlen) * "  window=" * string(windows[xt,:]) * "  len_win=" * string(len_win) * "  len_step=" * string(len_step)))
        MIs[xt] = MIxy(xsorted[(windows[xt,1]):(windows[xt,2])], ysorted[(windows[xt,1]):(windows[xt,2])])
    end
    maxMI, maxp = findmax(MIs)
    #
    return maxMI, windows[maxp,:]
end



#20230515

### (OPTIONAL) Optimize MI may appear in local samples
####
function localMIPinnedPropSample(xsorted::Union{AbstractVector{Float64},AbstractVector{Int64}}, ysorted::Union{AbstractVector{Float64},AbstractVector{Int64}},
                                prop_sample::Float64=0.6, step_window::Float64=5.0/length(xsorted))
    #
    xlen = length(xsorted)
    len_step = round(Int64, xlen * step_window)
    len_win = round(Int64, prop_sample * xlen)
    if len_step < 1
        len_step = 1
    end
    #
    n_step = floor(Int64, ((1.0 - prop_sample) * xlen / len_step))# "round" can results errors, " + 1" is not needed. It must use "/ len_step".
    # Check n_step
    stepsTox = len_win + (n_step - 1) * len_step
    # println(string(stepsTox) * " " * string(xlen) * " " * string(len_step) * " " * string(len_win))
    if stepsTox < xlen# && 
        n_step = n_step + 1
    end
    # Windows for n steps
    windows = zeros(Int64, n_step, 2)
    len_step < len_win || throw(ArgumentError("prop_sample is too small"))
    Threads.@threads for xt = 1:n_step
        windows[xt, :] .= [Int64(1 + (xt - 1) * len_step), Int64(len_win + (xt - 1) * len_step)]
    end
    windows[n_step, :] .= [(xlen - len_win + 1), xlen]
    MIs = zeros(Float64, n_step)
    Threads.@threads for xt = 1:n_step
        MIs[xt] = MIxy(view(xsorted, (windows[xt,1]):(windows[xt,2])),
                       view(ysorted, (windows[xt,1]):(windows[xt,2])))
    end
    maxMI, maxp = findmax(MIs)
    return maxMI, windows[maxp,:]
end
####
function optimLocalMI(X::Union{AbstractVector{Float64},AbstractVector{Int64}}, Y::Union{AbstractVector{Float64},AbstractVector{Int64}},
                    min_sample_prop::Float64=0.6, step_prop::Float64=0.05, step_window::Float64=5.0/length(X);
                    getOnlyOptim::Bool=true, skipCheck::Bool=true)
    skipCheck || 0.0 < min_sample_prop < 1.0 || throw(ArgumentError("min_sample_prop must in (0, 1)"))
    n_prop = floor(Int64, (1.0 - min_sample_prop) / step_prop)# exclude 1.0
    proportions = zeros(Float64, n_prop)
    Threads.@threads for xp = 1:n_prop
        proportions[xp] = min_sample_prop + (xp - 1) * step_prop
    end
    if !skipCheck
        if !((1.0 / length(X)) ≤ step_window < 0.5)
            step_window = 1.0 / length(X)
        end
    end
    MI_full = MIxy(X, Y)
    xsortp = sortperm(X)
    xsorted = X[xsortp]
    ysorted = Y[xsortp]
    #
    props = zeros(Int64, n_prop, 2)
    MIs = zeros(Float64, n_prop)
    for xp in eachindex(proportions)
        xm, xw = localMIPinnedPropSample(xsorted, ysorted, proportions[xp], step_window)
        props[xp, :] .= xw
        MIs[xp] = xm
    end
    max_local_MI, max_posi = findmax(MIs)
    max_MI = maximum([max_local_MI, MI_full])
    if getOnlyOptim
        return max_MI
    else
        global_or_local = max_local_MI < MI_full
        local_best_prop = proportions[max_posi]
        local_best_samples = [x for x in 1:length(X)][xsortp][(props[max_posi,1]):(props[max_posi,2])] |> sort
        out = myT_LocalOptimMI(max_MI, global_or_local, MI_full, max_local_MI, local_best_prop, local_best_samples)
        return out
    end
end
# function optimLocalMI(X::Union{AbstractVector{Float64},AbstractVector{Int64}}, Y::Union{AbstractVector{Float64},AbstractVector{Int64}},
#                     min_sample_size::Int64, step_size::Int64=50)

# end


# 20230516


### matrix modifier
function atomMI!(mat::AbstractMatrix{Float64},
                xr::Int64, xc::Int64,
                xr_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}}, xc_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64,
                rm_outliers::Bool=true)
    #
    anynan = false
    if any(isnan, xr_vec)
        mat[xr, :] .= NaN
        anynan = true
    end
    if any(isnan, xc_vec)
        mat[xr,xc] = NaN
        anynan = true
    end
    if !anynan
        if rm_outliers
            xro, xco = rmOutliersVecXY(xr_vec, xc_vec)
            mat[xr,xc] = optimLocalMI(xro, xco, false, slideStep, widthRange, widthStep)# or MIxy
        else
            mat[xr,xc] = optimLocalMI(xr_vec, xc_vec, false, slideStep, widthRange, widthStep)# or MIxy
        end
    end
end


### MI for X-X. Normalization for each feature. UpperTriangular makes it faster.
function calcMITriangle(X::Union{AbstractMatrix{Float64}, AbstractDataFrame}, rm_outliers::Bool=true, nan_or_zero::Bool=true,
                        min_sample_prop::Float64=0.6, step_prop::Float64=0.05, step_window::Float64=5.0/size(X,1))
    nc = size(X, 2)# col: features
    C = zeros(Float64, nc, nc)
    # combn
    combn_vec = collect(combinations(1:nc, 2))
    # Calc MI
    Threads.@threads for cx in eachindex(combn_vec)
        atomMI!(C,
                combn_vec[cx][1], combn_vec[cx][2],
                view(X, :, combn_vec[cx][1]), view(X, :, combn_vec[cx][2]),
                min_sample_prop, step_prop, step_window, rm_outliers)
    end
    if nan_or_zero
        Threads.@threads for xr = 1:nc
            C[xr, 1:xr] .= NaN
        end
    end
    return C
end


# 20230522
### matrix modifier
function atomMI(xr_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}}, xc_vec::Union{AbstractVector{Float64}, AbstractVector{Int64}},
                slideStep::Union{Int64, Float64}, widthRange::Vector{Float64}, widthStep::Float64,
                rm_outliers::Bool=true, outlx::Vector{Int64}=detectOutliers(xr_vec), outly::Vector{Int64}=detectOutliers(xc_vec))
    #
    if rm_outliers
        xro, xco, wh2save = rmOutliersVecXY(xr_vec, xc_vec, outlx, outly, false)
        optimumxs = optimLocalMI(xro, xco, false, slideStep, widthRange, widthStep)
        if !(optimumxs.global_or_local)
            wh2save = intersect(wh2save, optimumxs.local_samples)
        end
        if length(xr_vec) == length(wh2save)
            wh2save = Int64[]# Means all samples are saved
        end
    else
        optimumxs = optimLocalMI(xr_vec, xc_vec, false, slideStep, widthRange, widthStep)
        if optimumxs.global_or_local
            wh2save = Int64[]
        else
            wh2save = optimumxs.local_samples
        end
    end
    return optimumxs.MI_max, wh2save
end

