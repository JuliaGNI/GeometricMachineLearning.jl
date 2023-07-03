using GeometricMachineLearning: tensor_mat_mul
using Random, Test

backend = backend = KernelAbstractions.CPU()

a = rand!(allocate(backend, Float32, 256, 123, 10))
b = rand!(allocate(backend, Float32, 123, 45))
c = KernelAbstractions.zeros(backend, Float32, 256, 45, 10)

tensor_mat_mul!(a,b,c)
KernelAbstractions.synchronize(backend)

c_manual = zeros(Float32, dim1, dim3, num_data)
@time for i in 1:num_data
    c_manual[:,:,i] = a[:,:,i]*b 
end

@test isapprox(c, c_manual)
