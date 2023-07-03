# Simple kernel for tensor-matrix multiplication (maybe you need to add a block index here!)
@kernel function tensor_mat_mul_kernel!(c, a, b)
    i, j, k = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for l = 1:size(a)[2]
        tmp_sum += a[i, l, k] * b[l, j]
    end

    c[i,j,k] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function tensor_mat_mul!(c, a, b)
    @assert size(a)[2] == size(b)[1]

    backend = KernelAbstractions.get_backend(a)
    kernel! = tensor_mat_mul_kernel!(backend)
    kernel!(c, a, b, ndrange=size(c)) 
end