# Simple kernel for tensor-matrix multiplication (maybe you need to add a block index here!)
@kernel function tensor_mat_mul_kernel!(C, A, B)
    i, j, k = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(C))
    for l = 1:size(A)[2]
        tmp_sum += A[i, l, k] * B[l, j]
    end

    C[i,j,k] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function tensor_mat_mul!(C, A, B)
    @assert size(A)[2] == size(B)[1]

    backend = KernelAbstractions.get_backend(A)
    kernel! = tensor_mat_mul_kernel!(backend)
    kernel!(C, A, B, ndrange=size(C)) 
end

function tensor_mat_mul(A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T 
    sizeA = size(A); sizeB = size(B)
    @assert sizeA[2] == sizeB[1] 
    tensor_shape = (sizeA[1], sizeB[2], sizeA[3])
    backend = get_backend(A)
    C = KernelAbstractions.zeros(backend, T, tensor_shape...)
    tensor_mat_mul!(C, A, B)
    C
end

########################### SymmetricMatrix (right multiplication)

@kernel function symmetric_mat_right_mul_kernel!(C::AbstractArray{T, 3}, B::AbstractArray{T, 3}, S::AbstractVector{T}, n::Int) where T
    i, j, l = @index(Global, NTuple)
    tmp_sum = zero(T)
    
    for k = j:n
        tmp_sum += B[i, k] * S[(k - 1)* k ÷ 2 + j]
    end
    
    for k = 1:(j - 1)
        tmp_sum += B[i, k] * S[(j - 1) * j ÷ 2 + k]
    end

    C[i, j, l] = tmp_sum
end

function symmetric_mat_right_mul!(C::AbstractArray{T, 3}, B::AbstractArray{T, 3}, S::AbstractVector{T}, n::Int) where T
    backend = KernelAbstractions.get_backend(C)
    
    symmetric_mat_right_mul_k! = symmetric_mat_right_mul_kernel!(backend)
    symmetric_mat_right_mul_k!(C, S, B, n, ndrange = size(C))

    nothing
end

function symmetric_mat_right_mul(B::AbstractArray{T, 3}, S::AbstractVector{T}, n::Int) where T
    C = copy(B)

    symmetric_mat_right_mul!(C, B, S, n)

    C
end

function tensor_mat_mul!(C::AbstractArray{T, 3}, B::AbstractArray{T, 3}, A::SymmetricMatrix{T}) where T
    @assert A.n == size(C, 2) == size(B, 2)

    symmetric_mat_mul!(C, B, A.S, A.n)
end