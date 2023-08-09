"""
This computes the "exponential" of a tensor. This is done by first transforming the tensor to a sparse matrix and then applying the exp function."
"""

@kernel function assign_matrix_from_tensor_kernel!(B::AbstractMatrix{T}, A::AbstractArray{T, 3}, m) where T
    i,j,k = @index(Global, NTuple)
    B[i+m*(k-1),j + m*(k-1)] = A[i,j,k]
end

function assign_matrix_from_tensor(A::AbstractArray{T, 3}) where T
    m, m2, batch_size = size(A)
    @assert m == m2 
    backend = KernelAbstractions.get_backend(A)
    B = KernelAbstractions.allocate(backend, T, m*batch_size, m*batch_size)
    assign_matrix_from_tensor! = assign_matrix_from_tensor_kernel!(backend)
    assign_matrix_from_tensor!(B, A, m, ndrange=size(A))
    B
end


@kernel function assign_tensor_from_matrix_kernel!(A::AbstractArray{T, 3}, B::AbstractMatrix{T}, m) where T 
    i,j,k = @index(Global, NTuple)
    A[i,j,k] = B[i+m*(k-1),j+m*(k-1)]
end

function assign_tensor_from_matrix(B::AbstractMatrix{T}, m, batch_size) where T
    backend = KernelAbstractions.get_backend(B)
    assign_tensor_from_matrix! = assign_tensor_from_matrix_kernel!(backend)
    A = KernelAbstractions.allocate(backend, T, m, m, batch_size)
    assign_tensor_from_matrix!(A, B, m, ndrange=size(A))
    A
end

function tensor_exponential(A::AbstractArray{T, 3}) where T 
    m, m2, batch_size = size(A)
    @assert m == m2

    full_matrix = assign_matrix_from_tensor(A)
    expA = exp(full_matrix)

    assign_tensor_from_matrix(expA, m, batch_size)
end
