using GeometricMachineLearning
using GeometricMachineLearning: tensor_mat_mul!, tensor_mat_mul, allocate
import KernelAbstractions
using Random, Test

Random.seed!(123)

backend = CPU()

const dim1 = 5
const dim2 = 12
const dim3 = 10
const num_data = 100

a = rand!(allocate(backend, Float32, dim1, dim2, num_data))
b = rand!(allocate(backend, Float32, dim2, dim3))
c = KernelAbstractions.zeros(backend, Float32, dim1, dim3, num_data)
tensor_mat_mul!(c, a, b)

c_manual = KernelAbstractions.zeros(backend, Float32, dim1, dim3, num_data)
for i in 1:num_data
    @views c_manual[:,:,i] = a[:,:,i] * b 
end

@test isapprox(c, c_manual)

function test_tensor_multiplication(first_dim::Int=dim1, second_dim::Int=dim2, third_dim::Int=dim3; T = Float64)
    A = rand(SymmetricMatrix{T}, second_dim)
    B = rand(first_dim, second_dim, third_dim)
    BA = tensor_mat_mul(B, A)
    for l in 1:third_dim
        @test (@view BA[:, :, l]) ≈ (@view B[:, :, l]) * A
    end
end

test_tensor_multiplication()