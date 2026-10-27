using Test, KernelAbstractions, GeometricMachineLearning
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

@doc raw"""
This function tests the setup of the transformer with Stiefel weights.
"""
function transformer_setup_test(dim, n_heads, L, T)
    model = Transformer(dim, n_heads, L, Stiefel = true)
    ps = NeuralNetwork(model, KernelAbstractions.CPU(), T).params
    @test typeof(ps[1].PQ.head_1) <: StiefelManifold
    @test eltype(ps[1].PQ.head_1) === T
end

for T in REAL_ELTYPES
    transformer_setup_test(10, 5, 4, T)
end
