using KernelAbstractions, Test, Random
#include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# Simple kernel for matrix multiplication
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

backend = KernelAbstractions.CPU()

#a = rand!(allocate(backend, Float32, 256, 123, 10))
#b = rand!(allocate(backend, Float32, 123, 45))
#c = KernelAbstractions.zeros(backend, Float32, 256, 45, 10)

dim1 = 2^4
dim2 = 2^1
dim3 = 2^4
num_data = 2^20

a = rand(Float32, dim1, dim2, num_data)
b = rand(Float32, dim2, dim3)
c = zeros(Float32, dim1, dim3, num_data)

@time tensor_mat_mul!(c,a,b)
KernelAbstractions.synchronize(backend)

c_manual = zeros(Float32, dim1, dim3, num_data)
@time for i in 1:num_data
    c_manual[:,:,i] = a[:,:,i]*b 
end

isapprox(c, c_manual)

