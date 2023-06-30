using GeometricMachineLearning: tensor_mat_mul

a = rand!(allocate(backend, Float32, 256, 123, 10))
b = rand!(allocate(backend, Float32, 123, 45))
c = KernelAbstractions.zeros(backend, Float32, 256, 45, 10)

tensor_mat_mul!(a,b,c)
KernelAbstractions.synchronize(backend)

#a_mul_b = 

#@test isapprox(c, a*b)
