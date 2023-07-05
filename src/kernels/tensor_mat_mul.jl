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