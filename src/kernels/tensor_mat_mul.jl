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

@doc raw"""
    tensor_mat_mul!(C, A, B)

Multiply the matrix `B` onto the tensor `A` from the right and store the result in `C`.

The function [`tensor_mat_mul`](@ref) calls `tensor_mat_mul!` internally.
"""
function tensor_mat_mul!(C::AbstractArray{<:Number, 3}, A::AbstractArray{<:Number, 3}, B::AbstractMatrix)
    @assert size(A)[2] == size(B)[1]

    backend = KernelAbstractions.get_backend(A)
    kernel! = tensor_mat_mul_kernel!(backend)
    kernel!(C, A, B, ndrange=size(C)) 
end

@doc raw"""
    tensor_mat_mul(A::AbstractArray{<:Number, 3}, B::AbstractMatrix)

Multipliy the matrix `B` onto the tensor `A` from the right. 

Internally this calls the inplace version [`tensor_mat_mul!`](@ref).

# Examples

```jldoctest
using GeometricMachineLearning: tensor_mat_mul

A = [1 1 1; 1 1 1; 1 1 1;;; 2 2 2; 2 2 2; 2 2 2]
B = [3 0 0; 0 2 0; 0 0 1]

tensor_mat_mul(A, B)

# output

3×3×2 Array{Int64, 3}:
[:, :, 1] =
 3  2  1
 3  2  1
 3  2  1

[:, :, 2] =
 6  4  2
 6  4  2
 6  4  2
```
"""
function tensor_mat_mul(A::AbstractArray{<:Number, 3}, B::AbstractMatrix)
    @assert eltype(A) == eltype(B)
    T = eltype(A)
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
        tmp_sum += B[i, k, l] * S[(k - 1)* k ÷ 2 + j]
    end
    
    for k = 1:(j - 1)
        tmp_sum += B[i, k, l] * S[(j - 1) * j ÷ 2 + k]
    end

    C[i, j, l] = tmp_sum
end

function symmetric_mat_right_mul!(C::AbstractArray{T, 3}, B::AbstractArray{T, 3}, S::AbstractVector{T}, n::Int) where T
    backend = KernelAbstractions.get_backend(C)
    
    symmetric_mat_right_mul_k! = symmetric_mat_right_mul_kernel!(backend)
    symmetric_mat_right_mul_k!(C, B, S, n, ndrange = size(C))

    nothing
end

function symmetric_mat_right_mul(B::AbstractArray{T, 3}, S::AbstractVector{T}, n::Int) where T
    C = copy(B)

    symmetric_mat_right_mul!(C, B, S, n)

    C
end

@doc raw"""
    mat_tensor_mul!(C::AbstractArray{<:Number, 3}, B::AbstractArray{<:Number, 3}, A::SymmetricMatrix)

Multiply the symmetric matrix `A` onto the tensor `B` from the right and store the result in `C`.

This performs an efficient multiplication based on the special structure of the symmetric matrix `A`.
"""
function tensor_mat_mul!(C::AbstractArray{<:Number, 3}, B::AbstractArray{<:Number, 3}, A::SymmetricMatrix)
    @assert eltype(C) == eltype(B) == eltype(A)
    @assert A.n == size(C, 2) == size(B, 2)

    symmetric_mat_right_mul!(C, B, A.S, A.n)
end