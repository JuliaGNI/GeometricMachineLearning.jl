"""
This implements the operation (A,B) -> A'*B for two tensors
"""

# Simple kernel for tensor-matrix multiplication (maybe you need to add a block index here!)
@kernel function tensor_transpose_tensor_mul_kernel!(c, a, b)
    i, j, k = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for l = 1:size(a)[2]
        tmp_sum += a[l, i, k] * b[l, j, k]
    end

    c[i,j,k] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function tensor_transpose_tensor_mul!(c, a, b)
    @assert size(a)[3] == size(b)[3]
    @assert size(a)[1] == size(b)[1]

    backend = KernelAbstractions.get_backend(a)
    kernel! = tensor_transpose_tensor_mul_kernel!(backend)
    kernel!(c, a, b, ndrange=size(c)) 
end