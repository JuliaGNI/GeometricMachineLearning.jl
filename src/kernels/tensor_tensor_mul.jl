# Simple kernel for tensor-matrix multiplication (maybe you need to add a block index here!)
@kernel function tensor_tensor_mul_kernel!(c, a, b)
    i, j, k = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for l = 1:size(a)[2]
        tmp_sum += a[i, l, k] * b[l, j, k]
    end

    c[i,j,k] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function tensor_tensor_mul!(c, a, b)
    @assert size(a)[3] == size(b)[3]
    @assert size(a)[2] == size(b)[1]

    backend = networkbackend(a)
    kernel! = tensor_tensor_mul_kernel!(backend)
    kernel!(c, a, b, ndrange=size(c)) 
end

function tensor_tensor_mul(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    backend = networkbackend(A)
    C = KernelAbstractions.zeros(backend, T, size(A)[1], size(B)[2], size(A)[3])
    tensor_tensor_mul!(C, A, B)
    C
end