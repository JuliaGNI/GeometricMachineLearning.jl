using Test, KernelAbstractions, GeometricMachineLearning

@doc raw"""
This function tests the setup of the transformer with Stiefel weights.
"""
function transformer_setup_test(dim, n_heads, L, T)
    model = Transformer(dim, n_heads, L, Stiefel=true)
    ps = initialparameters(KernelAbstractions.CPU(), T, model)
    @test typeof(ps[1].PQ.head_1) <: StiefelManifold
end

transformer_setup_test(10, 5, 4, Float32)