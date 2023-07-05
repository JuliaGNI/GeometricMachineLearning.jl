using GeometricMachineLearning: tensor_mat_mul!
using Random, Test

using CUDA
backend = CUDABackend()

dim1 = 256
dim2 = 123
dim3 = 45
num_data = 1000

a = rand!(allocate(backend, Float32, dim1, dim2, num_data))
b = rand!(allocate(backend, Float32, dim2, dim3))
c = KernelAbstractions.zeros(backend, Float32, dim1, dim3, num_data)

@time tensor_mat_mul!(c,a,b)
KernelAbstractions.synchronize(backend)

c_manual = KernelAbstractions.zeros(backend, Float32, dim1, dim3, num_data)
@time for i in 1:num_data
    c_manual[:,:,i] = a[:,:,i]*b 
end

@test isapprox(c, c_manual)
