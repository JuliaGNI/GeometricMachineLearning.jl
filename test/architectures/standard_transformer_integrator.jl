using GeometricMachineLearning
using Test
import Random

Random.seed!(123)

const sys_dim = 5
const model = StandardTransformerIntegrator(sys_dim)
const seq_length = 5

function set_up_and_apply_integrator(; T = Float32)
    ics = rand(T, sys_dim, seq_length)

    nn = NeuralNetwork(model, CPU(), T)

    iterates = iterate(nn, ics)

    @test ics ≉ iterates[:, (end - seq_length + 1):end]
end

set_up_and_apply_integrator(; T = Float32)
set_up_and_apply_integrator(; T = Float64)

@testset "upscaling_activation is stored under its name and reaches the upscaling layer ($T)" for T in (
    Float32, Float64)
    @test hasfield(StandardTransformerIntegrator, :upscaling_activation)

    arch = StandardTransformerIntegrator(4; transformer_dim = 6, n_heads = 2,
        upscaling_activation = tanh)
    @test arch.upscaling_activation === tanh

    # `Dense` stores `tanh` as AbstractNeuralNetworks' `TanhActivation()`, so the comparison is with
    # the activation that `Dense` makes of `tanh`, and against the one it makes of the default.
    upscaling = Chain(arch).layers[1]
    @test upscaling isa GeometricMachineLearning.Dense
    @test upscaling.σ === GeometricMachineLearning.Dense(4, 6, tanh).σ
    @test upscaling.σ !== GeometricMachineLearning.Dense(4, 6, identity).σ

    nn = NeuralNetwork(arch, CPU(), T)
    @test eltype(nn(rand(T, 4, 3))) === T
end
