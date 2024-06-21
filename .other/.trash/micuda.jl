include("act_env.jl")

# ============================================
# using StatsBase: iqr
using Statistics
using CausalityTools: mutualinfo, ValueHistogram, RectangularBinning, GaoKannanOhViswanath, MIShannon

## Estimate mutual information (MI) between x and y using the entropy/probability estimator RectangularBinning (the adaptive partitioning approach).
### Freedman-Diaconis' rule (no assumption on the distribution).
function binsFD(v)
    q1, q3 = quantile(v, [0.25, 0.75])
    bin_width = 2 * (q3 - q1) * length(v)^(-1/3)
    FD_float = (maximum(v) - minimum(v)) / bin_width
    if isnan(FD_float) || FD_float < 3 || FD_float > length(v)
        num_bins = round(Int64, length(v) * 0.1)
    else
        num_bins = round(Int64, FD_float)
    end
    return num_bins
end
### MI quantifies the "amount of information". The minimum is 0. Larger is "better".
function MIxy(X, Y;
              bins = binsFD.([X,Y]))
    est = ValueHistogram(RectangularBinning(bins))
    MI = mutualinfo(est, X, Y)
    return MI
end
# ============================================


# =
import CausalityTools: marginal_entropies_mi3h, SymbolicPermutation, Dispersion, marginal_encodings, StateSpaceSet, MutualInformation, entropy
# estimate(est::DifferentialEntropyEstimator, x, y) = estimate(MIShannon(), est, x, y)
# /home/malab13/.julia/packages/CausalityTools/DuYQs/src/methods/infomeasures/mutualinfo/MIShannon.jl
const WellDefinedMIShannonProbEsts{m, D} = Union{
    SymbolicPermutation{m},
    ValueHistogram{<:FixedRectangularBinning{D}},
    ValueHistogram{<:RectangularBinning{T}},
    Dispersion
} where {m, D, T}

function marginal_entropies_mi3h(est::WellDefinedMIShannonProbEsts{m, D}, x, y) where {m, D}
    measure::MutualInformation = MIShannon()
    eX, eY = marginal_encodings(est, x, y)
    eXY = StateSpaceSet(eX, eY)
    e = measure.e
    HX = entropy(e, CountOccurrences(), eX)
    HY = entropy(e, CountOccurrences(), eY)
    HXY = entropy(e, CountOccurrences(), eXY)
    return HX, HY, HXY
end

HX, HY, HXY = marginal_entropies_mi3h(measure, est, x, y)
mi = HX + HY - HXY

# = #


# ==============================================
# generate some sample data on the CPU
x_array = rand(Float64, 128)
y_array = x_array .+ rand(Float64, 128) / 100

# copy data to the GPU as CuArrays
using CUDA
x_d = CuArray(x_array)
y_d = CuArray(y_array)

# The two algor are the same ?
MIxy(x_array, y_array)
mutual_information(x_array, y_array)

# MIxy(x_d, y_d)

length(x_d)
@time sort(x_d)
@time sort(y_d)
@time sort(y_array)
y_d[7]
# binsFD(x_d)
# ==============================================


function mi_kernel(x_array::AbstractArray{Float64}, y_array::AbstractArray{Float64})
    # get thread index
    i = threadIdx().x
    # get variables from arrays
    x = x_array[i]
    y = y_array[i]
    # compute mutual information
    mi = mutual_info(x, y)
    # print result
    @cuprint("Mutual information between x[$i] and y[$i] is $mi\\n")
    return nothing
end


# launch kernel with 10 threads, one for each pair of variables
@cuda threads=10 mi_kernel(x_d, y_d)

# check output (may take a while to appear)
CUDA.synchronize()
