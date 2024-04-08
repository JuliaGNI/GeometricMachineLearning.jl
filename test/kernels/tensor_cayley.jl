using GeometricMachineLearning: tensor_cayley2, tensor_cayley3, tensor_cayley4, tensor_cayley5
using GeometricMachineLearning: tensor_transpose
using Test
import Random 

Random.seed!(123)

function check_orthonormal_property(B::Array)
    for i in axes(B, 3)
        B_temp = @view B[:, :, i]
        @test B_temp * B_temp' ≈ one(B_temp)
    end
end

function test_tensor_cayley2(T::Type=Float32, third_dim::Int=10)
    A = rand(2, 2, third_dim)
    B = tensor_cayley2(A - tensor_transpose(A))
    check_orthonormal_property(B)
end

function test_tensor_cayley3(T::Type=Float32, third_dim::Int=10)
    A = rand(3, 3, third_dim)
    B = tensor_cayley3(A - tensor_transpose(A))
    check_orthonormal_property(B)
end

function test_tensor_cayley4(T::Type=Float32, third_dim::Int=10)
    A = rand(4, 4, third_dim)
    B = tensor_cayley4(A - tensor_transpose(A))
    check_orthonormal_property(B)
end

function test_tensor_cayley5(T::Type=Float32, third_dim::Int=10)
    A = rand(5, 5, third_dim)
    B = tensor_cayley5(A - tensor_transpose(A))
    check_orthonormal_property(B)
end

function check_all(T::Type, third_dim::Int=10)
    test_tensor_cayley2(T, third_dim)
    test_tensor_cayley3(T, third_dim)
    test_tensor_cayley4(T, third_dim)
    test_tensor_cayley5(T, third_dim)
end

check_all(Float32)
check_all(Float64)