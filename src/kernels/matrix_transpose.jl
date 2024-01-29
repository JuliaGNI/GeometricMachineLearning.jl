"""
This computes the transpose of a matrix.
"""

@kernel function matrix_transpose_kernel!(C, A)
    i, j = @index(Global, NTuple)

    C[i, j] = A[j, i]
end

function matrix_transpose!(C, A)
    sizeA = size(A)
    sizeC = size(C)
    @assert sizeA[1] == sizeC[2]
    @assert sizeA[2] == sizeC[1]

    backend = KernelAbstractions.get_backend(A)
    kernel! = matrix_transpose_kernel!(backend)
    kernel!(C, A, ndrange=size(C))
end

function matrix_transpose(A::AbstractArray{T, 2}) where T 
    sizeA = size(A)
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.zeros(backend, T, sizeA[2], sizeA[1])
    matrix_transpose!(C, A)
    C 
end