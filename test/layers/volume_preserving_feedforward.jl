using GeometricMachineLearning
using LinearAlgebra: det
using Zygote: jacobian
using Test 

function test_volume_preservation(layer::GeometricMachineLearning.AbstractExplicitLayer, ps::NamedTuple, b::AbstractVector{T}) where T
    jac_mat = jacobian(b -> layer(b, ps), b)[1]
    @test det(jac_mat) ≈ one(T)
end

function test_volume_preserving_feedforward(dim₁ = 5; T::Type=Float32)
    layer₁ = VolumePreservingLowerLayer(dim₁; use_bias = false)
    layer₂ = VolumePreservingLowerLayer(dim₁; use_bias = true)
    layer₃ = VolumePreservingUpperLayer(dim₁; use_bias = false)
    layer₄ = VolumePreservingUpperLayer(dim₁; use_bias = true)

    ps₁ = NeuralNetwork(Chain(layer₁), CPU(), T).params.L1
    ps₂ = NeuralNetwork(Chain(layer₂), CPU(), T).params.L1
    ps₃ = NeuralNetwork(Chain(layer₃), CPU(), T).params.L1
    ps₄ = NeuralNetwork(Chain(layer₄), CPU(), T).params.L1

    # test if application to matrix and tensor gives same result
    test_vector = rand(T, dim₁)
    test_volume_preservation(layer₁, ps₁, test_vector)
    test_volume_preservation(layer₂, ps₂, test_vector)
    test_volume_preservation(layer₃, ps₃, test_vector)
    test_volume_preservation(layer₄, ps₄, test_vector)
end

test_volume_preserving_feedforward(; T = Float32)
test_volume_preserving_feedforward(; T = Float64)