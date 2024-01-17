using GeometricMachineLearning: symplectic_transformer_potential, symplectic_transformer_potential_gradient
using Test 
using Zygote

"""
Checks if we really compute the gradient of the potential (for a matrix).
"""
function test_gradient(T::DataType=Float32, n::Int=2, time_steps::Int=10)
    A = rand(SymmetricMatrix{T}, n)
    X = rand(T, n, time_steps)
    @test Zygote.gradient(X -> symplectic_transformer_potential(X, A), X)[1] ≈ symplectic_transformer_potential_gradient(X, A) 
end 

"""
Checks if we really compute the gradient of the potential (for a tensor).
"""
function test_gradient_tensor(T::DataType=Float32, n::Int=2, time_steps::Int=10, third_axis::Int=5)
    A = rand(T, n, n)
    X = rand(T, n, time_steps, third_axis)
    for f ∈ 1:third_axis
        display(@test Zygote.gradient(X -> symplectic_transformer_potential(X, A), X[:, :, f])[1] ≈ symplectic_transformer_potential_gradient(X, A)[:, :, f])
    end
end

test_gradient()
test_gradient_tensor()