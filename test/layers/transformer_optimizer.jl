using Test, KernelAbstractions, GeometricMachineLearning, Zygote, LinearAlgebra
using GeometricMachineLearning: ResNetLayer
using GeometricOptimizers: optimization_step!
import Random

Random.seed!(1234)

@doc raw"""
This function tests if the `GradientMethod`, `MomentumMethod`, and `Adam` act on the neural network weights via `optimization_step!`.
"""
function transformer_gradient_test(T, dim, n_heads, L, seq_length = 8, batch_size = 10)
    model = Chain(Transformer(dim, n_heads, L, Stiefel = true), ResNetLayer(dim))
    model = Transformer(dim, n_heads, L, Stiefel = true)

    ps = NeuralNetwork(model, KernelAbstractions.CPU(), T).params

    input = rand(T, dim, seq_length, batch_size)

    loss(ps, input) = norm(model(input, ps))
    dx = Zygote.gradient(ps -> loss(ps, input), ps)[1]

    ps₁ = deepcopy(ps)
    ps₂ = deepcopy(ps)
    ps₃ = deepcopy(ps)

    o₁ = NetworkOptimizer(ps₁; algorithm = GradientMethod())
    o₂ = NetworkOptimizer(ps₂; algorithm = MomentumMethod())
    o₃ = NetworkOptimizer(ps₃; algorithm = Adam())

    optimization_step!(ps₁, o₁, dx)
    optimization_step!(ps₂, o₂, dx)
    optimization_step!(ps₃, o₃, dx)
    @test typeof(ps₁) == typeof(ps₂) == typeof(ps₃) == typeof(ps)
    @test ps₁[1].PQ.head_1 ≈ ps₂[1].PQ.head_1
    @test ps₁[1].PQ.head_1 ≉ ps₃[1].PQ.head_1
end

transformer_gradient_test(Float32, 10, 5, 4)
