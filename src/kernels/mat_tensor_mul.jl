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

# Creating a wrapper kernel for launching with error checks
function mat_tensor_mul!(C, A, B)
    @assert size(A)[2] == size(B)[1]

    backend = KernelAbstractions.get_backend(A)
    kernel! = mat_tensor_mul_kernel!(backend)
    kernel!(C, A, B, ndrange=size(C)) 
end

function mat_tensor_mul(A::AbstractMatrix{T}, B::AbstractArray{T, 3}) where T
    sizeA = size(A)
    sizeB = size(B)
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.zeros(backend, T, sizeA[1], sizeB[2], sizeB[3])
    mat_tensor_mul!(C, A, B)
    C
end