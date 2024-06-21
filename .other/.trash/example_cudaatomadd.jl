using CUDA

function kernel(a::CuDeviceArray, b::CuDeviceArray)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(a)
        # CUDA.atomic_add!(a, i, b[i])
        a[i] += b[i]
    end
    return
end

a = CuArray([1, 2, 3])
b = CuArray([4, 5, 6])
@cuda threads=3 kernel(a, b)

Array(a)


# New
using CUDA
using Statistics

function binsFD(v::Union{AbstractVector{Float64},AbstractVector{Int64}, CuArray})::Int64
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
    return num_bins
end

function hist_kernel(hist::CuDeviceMatrix, x::CuDeviceArray, y::CuDeviceArray)
    lenX = length(x)
    lenY = length(y)
    lenX == lenY ||
        throw(ArgumentError("number of elements in each array must match"))
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= lenX && j <= lenY
        # hist[Int32(x[i]) + 1, Int32(y[j]) + 1] += 1
        # x[i] += y[j]
    end
    return nothing
end


x = CUDA.rand(512)
y = CUDA.rand(512)
hist = CuArray{Int32}(undef, (binsFD(x), binsFD(y)))

@cuda threads=128 hist_kernel(hist, x, y)
x
