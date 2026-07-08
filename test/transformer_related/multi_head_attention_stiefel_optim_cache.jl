using GeometricMachineLearning, GeometricOptimizers, Test
import Random, LinearAlgebra

Random.seed!(1234)

@doc raw"""
This checks if the Adam cache was set up in the right way.
AdamCache fields (from GeometricOptimizers): x, g, δ, Δg, m₁, m₂, m̃₂, section.
The direction δ and moments m₁, m₂ should be zero-initialised.
"""
function check_adam_cache(C::GeometricOptimizers.OptimizerCache{T}, tol=T(10) * eps(T)) where T
    @test C isa AdamCache
    # direction and first/second moments are zero-initialised
    @test typeof(C.δ) <: StiefelLieAlgHorMatrix
    @test typeof(C.m₁) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(C.δ) < tol
    @test LinearAlgebra.norm(C.m₁) < tol
end
check_adam_cache(B::NamedTuple) = apply_toNT(check_adam_cache, B)

@doc raw"""
This checks if the momentum cache was set up in the right way.
MomentumCache fields (from GeometricOptimizers): x, g, δ, Δg, section.
The direction δ should be zero-initialised.
"""
function check_momentum_cache(C::GeometricOptimizers.OptimizerCache{T}, tol=T(10) * eps(T)) where T
    @test C isa MomentumCache
    @test typeof(C.δ) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(C.δ) < tol
end
check_momentum_cache(B::NamedTuple) = apply_toNT(check_momentum_cache, B)

@doc raw"""
This checks if the gradient cache was set up in the right way.
GradientCache fields (from GeometricOptimizers): x, g, δ, Δg, section.
"""
function check_gradient_cache(C::GeometricOptimizers.OptimizerCache{T}) where T
    @test C isa GradientCache
    @test hasproperty(C, :δ)
end
check_gradient_cache(B::NamedTuple) = apply_toNT(check_gradient_cache, B)

@doc raw"""
This checks if all the caches are set up in the right way for the `MultiHeadAttention` layer with Stiefel weights.
"""
function test_cache_setups_for_optimizer_for_multihead_attention_layer(T::Type, dim::Int, n_heads::Int)
    @assert dim % n_heads == 0
    model = Chain(MultiHeadAttention(dim, n_heads, Stiefel=true))
    ps = NeuralNetwork(model, CPU(), T).params

    o₁ = Optimizer(Adam(), ps)
    o₂ = Optimizer(MomentumMethod(), ps)
    o₃ = Optimizer(GradientMethod(), ps)

    check_adam_cache(o₁.cache)
    check_momentum_cache(o₂.cache)
    check_gradient_cache(o₃.cache)
end

test_cache_setups_for_optimizer_for_multihead_attention_layer(Float32, 64, 8)
