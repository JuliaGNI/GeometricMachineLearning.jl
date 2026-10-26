using GeometricMachineLearning, Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

"""
This test checks if the sympnets handle tensors the right way.
"""
function sympnet_tests(::Type{T}, N, N2 = 2*N, second_dim = 10, third_dim = 10) where {T}
    model₁ = Chain(LinearLayerQ(N), LinearLayerP(N))
    model₂ = Chain(ActivationLayerQ(N, tanh), ActivationLayerP(N, tanh))
    model₃ = Chain(GradientLayerQ(N, N2, tanh), GradientLayerP(N, N2, tanh))
    ps₁ = NeuralNetwork(model₁, CPU(), T).params
    ps₂ = NeuralNetwork(model₂, CPU(), T).params
    ps₃ = NeuralNetwork(model₃, CPU(), T).params

    # evaluate functions 
    x_vec = rand(T, N)
    x_mat = rand(T, N, second_dim)
    x_ten = rand(T, N, second_dim, third_dim)
    model₁(x_vec, ps₁)
    model₁(x_mat, ps₁)
    model₁(x_ten, ps₁)
    model₂(x_vec, ps₂)
    model₂(x_mat, ps₂)
    model₂(x_ten, ps₂)
    model₃(x_vec, ps₃)
    model₃(x_mat, ps₃)
    model₃(x_ten, ps₃)

    @test eltype(model₁(x_ten, ps₁)) === T
    @test eltype(model₂(x_ten, ps₂)) === T
    @test eltype(model₃(x_ten, ps₃)) === T

    @test isapprox(model₁(x_ten[:, :, 1], ps₁), model₁(x_ten, ps₁)[:, :, 1])
    @test isapprox(model₂(x_ten[:, :, 1], ps₂), model₂(x_ten, ps₂)[:, :, 1])
    @test isapprox(model₃(x_ten[:, :, 1], ps₃), model₃(x_ten, ps₃)[:, :, 1])
end

for T in REAL_ELTYPES
    sympnet_tests(T, 10)
end
