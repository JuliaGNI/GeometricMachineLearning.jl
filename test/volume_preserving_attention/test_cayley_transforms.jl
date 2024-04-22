using GeometricMachineLearning: tensor_cayley4, tensor_cayley3, cpu_tensor_cayley, tensor_transpose
using Test 

function test_orthonormal(A::AbstractMatrix)
    @test A' * A ≈ one(A)
end

function test_orthonormal(A::AbstractArray{T, 3}) where T 
    for i in axes(A, 3)
        A_temp = @view A[:, :, i]
        test_orthonormal(A_temp)
    end
end

function test_cayley_transforms(third_dim::Int = 10)
    A₄ = rand(4, 4, third_dim)
    A₃ = rand(3, 3, third_dim)
    # asymmetrize 
    A₄ = A₄ - tensor_transpose(A₄)
    A₃ = A₃ - tensor_transpose(A₃)
    test_orthonormal(tensor_cayley4(A₄))
    test_orthonormal(tensor_cayley3(A₃))
    test_orthonormal(cpu_tensor_cayley(A₄))
    test_orthonormal(cpu_tensor_cayley(A₃))
end

test_cayley_transforms()