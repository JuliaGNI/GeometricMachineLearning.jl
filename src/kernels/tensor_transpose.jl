"""
This computes the transpose of a tensor.
"""

@kernel function tensor_transpose_kernel!(C, A)
    i, j, k = @index(Global, NTuple)

    C[i, j, k] = A[j, i, k]
end

function tensor_transpose!(C, A)
    sizeA = size(A)
    sizeC = size(C)
    @assert sizeA[3] == sizeC[3]
    @assert sizeA[1] == sizeC[2]
    @assert sizeA[2] == sizeC[1]

    backend = networkbackend(A)
    kernel! = tensor_transpose_kernel!(backend)
    kernel!(C, A, ndrange=size(C))
end

function tensor_transpose(A::AbstractArray{T, 3}) where T 
    sizeA = size(A)
    backend = networkbackend(A)
    C = KernelAbstractions.zeros(backend, T, sizeA[2], sizeA[1], sizeA[3])
    tensor_transpose!(C, A)
    C 
end