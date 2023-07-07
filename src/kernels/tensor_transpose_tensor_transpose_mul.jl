"""
This implements the operation (A,B) -> A'*B' for two tensors
"""

# Simple kernel for tensor-matrix multiplication (maybe you need to add a block index here!)
@kernel function tensor_transpose_tensor_transpose_mul_kernel!(C, A, B)
    i, j, k = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(C))
    for l = 1:size(A)[1]
        tmp_sum += A[l, i, k] * B[j, l, k]
    end

    C[i,j,k] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function tensor_transpose_tensor_transpose_mul!(C, A, B)
    @assert size(A)[3] == size(B)[3]
    @assert size(A)[1] == size(B)[2]

    backend = KernelAbstractions.get_backend(A)
    kernel! = tensor_transpose_tensor_transpose_mul_kernel!(backend)
    kernel!(C, A, B, ndrange=size(C)) 
end

function tensor_transpose_tensor_transpose_mul(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.zeros(backend, T, size(A)[2], size(B)[1], size(A)[3])
    tensor_transpose_tensor_transpose_mul!(C, A, B)
    C
end
