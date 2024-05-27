# Simple kernel for tensor-matrix multiplication (maybe you need to add a block index here!)
@kernel function mat_tensor_mul_kernel!(C, A, B)
    i, j, k = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(C))
    for l in axes(A, 2)
        tmp_sum += A[i, l] * B[l, j, k]
    end

    C[i,j,k] = tmp_sum

    nothing
end

@doc raw"""
    mat_tensor_mul!(C, A, B)

Multiply the matrix `A` onto the tensor `B` from the left and store the result in `C`.

Also checks the bounds of the input arrays.

The function [`mat_tensor_mul`](@ref) calls `mat_tensor_mul!` internally.
"""
function mat_tensor_mul!(C, A, B)
    @assert size(A)[2] == size(B)[1]

    backend = KernelAbstractions.get_backend(A)
    kernel! = mat_tensor_mul_kernel!(backend)
    kernel!(C, A, B, ndrange=size(C)) 
end

@doc raw"""
    mat_tensor_mul(A::AbstractMatrix{T}, B::AbstractArray{T, 3}) where T

Multipliy the matrix `A` onto the tensor `B` from the left. 

Internally this calls the inplace version [`mat_tensor_mul!`](@ref).

# Examples

```jldoctest
using GeometricMachineLearning: mat_tensor_mul

B = [1 1 1; 1 1 1; 1 1 1;;; 2 2 2; 2 2 2; 2 2 2]
A = [3 0 0; 0 2 0; 0 0 1]

mat_tensor_mul(A, B)

# output

3×3×2 Array{Int64, 3}:
[:, :, 1] =
 3  3  3
 2  2  2
 1  1  1

[:, :, 2] =
 6  6  6
 4  4  4
 2  2  2
```
"""
function mat_tensor_mul(A::AbstractMatrix{T}, B::AbstractArray{T, 3}) where T
    sizeA = size(A)
    sizeB = size(B)
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.zeros(backend, T, sizeA[1], sizeB[2], sizeB[3])
    mat_tensor_mul!(C, A, B)
    C
end

################### SymmetricMatrix

@kernel function symmetric_mat_mul_kernel!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    i, j, l = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = 1:i 
        tmp_sum += S[((i - 1) * i) ÷ 2 + k] * B[k, j, l]
    end
    for k = (i+1):n 
        tmp_sum += S[((k - 1) * k) ÷ 2 + i] * B[k, j, l]
    end
    C[i, j, l] = tmp_sum
end

function symmetric_mat_mul!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    backend = KernelAbstractions.get_backend(C)

    symmetric_mat_mul_k! = symmetric_mat_mul_kernel!(backend)
    symmetric_mat_mul_k!(C, S, B, n, ndrange=size(C))
end

function symmetric_mat_mul(S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    C = copy(B)

    symmetric_mat_mul!(C, S, B, n)

    C
end

@doc raw"""
    mat_tensor_mul!(C::AbstractArray{T, 3}, A::SymmetricMatrix{T}, B::AbstractArray{T, 3}) where T

Multiply the symmetric matrix `A` onto the tensor `B` from the left and store the result in `C`.

Also checks the bounds of the input arrays. 

This performs an efficient multiplication based on the special structure of the symmetric matrix `A`.
"""
function mat_tensor_mul!(C::AbstractArray{T, 3}, A::SymmetricMatrix{T}, B::AbstractArray{T, 3}) where T 
    @assert A.n == size(C, 1) == size(B, 1)

    symmetric_mat_mul!(C, A.S, B, A.n)
end

########################### LowerTriangular

@kernel function lo_mul_kernel!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, ::Int) where T
    i, j, l = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = 1:(i-1)
        tmp_sum +=  S[(i - 2) * (i - 1) ÷ 2 + k] * B[k, j, l]
    end
    C[i, j, l] = tmp_sum

    nothing
end

function lo_mat_mul!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    backend = KernelAbstractions.get_backend(C)

    lo_mat_mul_k! = lo_mul_kernel!(backend)
    lo_mat_mul_k!(C, S, B, n, ndrange=size(C))
end

function lo_mat_mul(S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    C = zero(B)
    
    lo_mat_mul!(C, S, B, n)

    C 
end

@doc raw"""
    mat_tensor_mul!(C::AbstractArray{T, 3}, A::LowerTriangular{T}, B::AbstractArray{T, 3}) where T

Multiply the lower-triangular matrix `A` onto the tensor `B` from the left and store the result in `C`.

Also checks the bounds of the input arrays. 

This performs an efficient multiplication based on the special structure of the lower-triangular matrix `A`.
"""
function mat_tensor_mul!(C::AbstractArray{T, 3}, A::LowerTriangular{T}, B::AbstractArray{T, 3}) where T
    @assert A.n == size(C, 1) == size(B, 1)

    lo_mat_mul!(C, A.S, B, A.n)
end

####################### UpperTriangular

@kernel function up_mul_kernel!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T
    i, j, l = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = (i+1):n 
        tmp_sum += S[(k - 2) * (k - 1) ÷ 2 + i] * B[k, j, l]
    end
    C[i, j, l] = tmp_sum

    nothing
end

function up_mat_mul!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    backend = KernelAbstractions.get_backend(C)

    up_mat_mul_k! = up_mul_kernel!(backend)
    up_mat_mul_k!(C, S, B, n, ndrange=size(C))
end

function up_mat_mul(S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    C = zero(B)

    up_mat_mul!(C, S, B, n)

    C
end

@doc raw"""
    mat_tensor_mul!(C::AbstractArray{T, 3}, A::UpperTriangular{T}, B::AbstractArray{T, 3}) where T

Multiply the upper-triangular matrix `A` onto the tensor `B` from the left and store the result in `C`.

Also checks the bounds of the input arrays. 

This performs an efficient multiplication based on the special structure of the upper-triangular matrix `A`.
"""
function mat_tensor_mul!(C::AbstractArray{T, 3}, A::UpperTriangular{T}, B::AbstractArray{T, 3}) where T 
    @assert A.n == size(C, 1) == size(B, 1)

    up_mat_mul!(C, A.S, B, A.n)
end


################## SkewSymMatrix

@kernel function skew_mat_mul_kernel!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T
    i, j, l = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = 1:(i-1)
        tmp_sum +=  S[(i - 2) * (i - 1) ÷ 2 + k] * B[k, j, l]
    end
    for k = (i+1):n 
        tmp_sum += -S[(k - 2) * (k - 1) ÷ 2 + i] * B[k, j, l]
    end
    C[i, j, l] = tmp_sum
end

function skew_mat_mul!(C::AbstractArray{T, 3}, S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    backend = KernelAbstractions.get_backend(C)

    skew_mat_mul_k! = skew_mat_mul_kernel!(backend)
    skew_mat_mul_k!(C, S, B, n, ndrange=size(C))
end

function skew_mat_mul(S::AbstractVector{T}, B::AbstractArray{T, 3}, n::Int) where T 
    C = zero(B)

    skew_mat_mul!(C, S, B, n)

    C
end

@doc raw"""
    mat_tensor_mul!(C::AbstractArray{T, 3}, A::SkewSymMatrix{T}, B::AbstractArray{T, 3}) where T

Multiply skew-symmetric the matrix `A` onto the tensor `B` from the left and store the result in `C`.

Also checks the bounds of the input arrays. 

This performs an efficient multiplication based on the special structure of the skew-symmetric matrix `A`.
"""
function mat_tensor_mul!(C::AbstractArray{T, 3}, A::SkewSymMatrix{T}, B::AbstractArray{T, 3}) where T
    @assert A.n == size(C, 1) == size(B, 1)

    skew_mat_mul!(C, A.S, B, A.n)
end