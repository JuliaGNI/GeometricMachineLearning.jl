using KernelAbstractions
using GeometricMachineLearning: tensor_mat_mul!

function tensor_mat_mul(A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T 
    sizeA = size(A); sizeB = size(B)
    @assert sizeA[2] == sizeB[1] 
    tensor_shape = (sizeA[1], sizeB[2], sizeA[3])
    backend = get_backend(A)
    C = KernelAbstractions.zeros(backend, tensor_shape...)
    tensor_mat_mul!(C, A, B)
    C
end


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
    @assert size(A)[1] = size(B)[1]

    backend = KernelAbstractions.get_backend(A)
    kernel! = tensor_transpose_mat_mul_kernel!(backend)
    kernel!(C, A, B, size(C))
end

function tensor_transpose_mat_mul(A, B)
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.zeros(backend, size(A)[2], size(B)[2], size(A)[3])
    tensor_transpose_mat_mul!(C, A, B)
    C
end

function tensor_transpose_tensor_mul(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.zeros(backend, size(A)[2], size(B)[2], size(A)[3])
    tenor_transpose_mat_mul!(C, A, B)
    C
end


#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(tensor_mat_mul), A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T
    sizeA = size(A); sizeB = size(B)
    @assert sizeA[2] == sizeB[1] 
    C = tensor_mat_mul(A, B)
    function tensor_mat_mul_pullback(C_diff)
        f̄ = NoTangent()
        #tensor_transpose_mat_mul
        A_diff = @thunk tensor_mat_mul(C_diff, B')
        B_diff = @thunk tensor_transpose_tensor_mul(A, C_diff)
        return f̄, f̄oo, b̄
    end
    return y, foo_mul_pullback
end
    
#rrule is compared to finite differences (FiniteDifferences.jl) 
using ChainRulesTestUtils
test_rrule(tensor_mat_mul, rand(3, 3, 3), rand(3, 3))
