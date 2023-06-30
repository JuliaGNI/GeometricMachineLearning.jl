using KernelAbstractions, Test, Random
#include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# Simple kernel for matrix multiplication
@kernel function tensor_mat_mul_kernel!(c, a, b)
    i, j, k = @index(Global, NTuple{3})

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for l = 1:size(a)[2]
        tmp_sum += a[i, l, k] * b[l, j]
    end

    c[i,j,k] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function tensor_mat_mul!(c, a, b)
    if size(a)[2] != size(b)[1]
        println("Tensor and matrix size mismatch!")
        return nothing
    end
    backend = KernelAbstractions.get_backend(a)
    kernel! = tensor_mat_mul_kernel!(backend)
    kernel!(a, b, c, ndrange=size(c)) 
end
