using Test, KernelAbstractions, GeometricMachineLearning
import Random

Random.seed!(1234)

@doc raw"""
This function tests the setup of the transformer with Stiefel weights.
"""
function transformer_setup_test(dim, n_heads, L, T)
    model = Transformer(dim, n_heads, L, Stiefel = true)
    ps = NeuralNetwork(model, KernelAbstractions.CPU(), T).params
    @test typeof(ps[1].PQ.head_1) <: StiefelManifold
end

transformer_setup_test(10, 5, 4, Float32)
