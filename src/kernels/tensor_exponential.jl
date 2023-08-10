"""
This computes the "exponential" of a tensor. This is done by first transforming the tensor to a sparse matrix and then applying the exp function."
"""
#=
@kernel function tensor_exponential_kernel!(B::AbstractArray{T, 3}, A::AbstractArray{T, 3}, m) where T
    i = @index(Global)
    B[:,:,i] = exp(A[:,:,i])
end

function tensor_exponential2(A::AbstractArray{T, 3}) where T 
    m, m2, batch_size = size(A)
    @assert m == m2
    backend = KernelAbstractions.get_backend(A)
    output = similar(A)
    tensor_exponential! = tensor_exponential_kernel!(backend)
    tensor_exponential!(output, A, batch_size)

    output
end
=#

function tensor_exponential(A::AbstractArray{T, 3}) where T
    output = zero(A)
    m, m2, batch_size = size(A)
    for k in axes(A, 3)
        B = exp(allocate_matrix(A, k))
        output += allocate_tensor(B, batch_size, k)
    end
    output 
end

@kernel function allocate_matrix_kernel!(B::AbstractMatrix{T}, A::AbstractArray{T, 3}, k) where T
    i,j = @index(Global, NTuple)
    B[i,j] = A[i,j,k]
end

function allocate_matrix(A::AbstractArray{T, 3}, k) where T
    sizeB = size(A)[1:2]
    backend = KernelAbstractions.get_backend(A)
    B = KernelAbstractions.allocate(backend, T, sizeB)
    allocate_matrix! = allocate_matrix_kernel!(backend)
    allocate_matrix!(B, A, k, ndrange=sizeB)
    B 
end

@kernel function allocate_tensor_kernel!(A::AbstractArray{T, 3}, B::AbstractMatrix{T}, k) where T 
    i,j = @index(Global, NTuple)
    A[i,j,k] = B[i,j]
end

function allocate_tensor!(A::AbstractArray{T,3}, B::AbstractMatrix{T}, k) where T
    backend = KernelAbstractions.get_backend(A)
    allocate_tensor! = allocate_tensor_kernel!(backend)
    allocate_tensor!(A, B, k, ndrange=size(B))
end

function allocate_tensor(B::AbstractMatrix{T}, batch_size, k) where T
    backend = KernelAbstractions.get_backend(B)
    A = KernelAbstractions.zeros(backend, T, size(B)..., batch_size)
    allocate_tensor!(A, B, k)
    A
end