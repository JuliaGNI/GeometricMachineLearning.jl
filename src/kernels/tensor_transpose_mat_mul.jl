"""
This implements the operation (A,B) -> A'*B for a tensor and a matrix
"""

@kernel function tensor_transpose_mat_mul_kernel!(C, A, B)
    i, j, k = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(C))
    for l = 1:size(A)[1]
        tmp_sum += A[l, i, k] * B[l, j]
    end

    C[i,j,k] = tmp_sum
end

function tensor_transpose_mat_mul!(C, A, B)
    @assert size(A)[1] == size(B)[1]

    backend = KernelAbstractions.get_backend(A)
    kernel! = tensor_transpose_mat_mul_kernel!(backend)
    kernel!(C, A, B, ndrange=size(C))
end

function tensor_transpose_mat_mul(A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.zeros(backend, T, size(A)[2], size(B)[2], size(A)[3])
    tensor_transpose_mat_mul!(C, A, B)
    C
end
