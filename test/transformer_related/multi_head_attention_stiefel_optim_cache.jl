using GeometricMachineLearning, Test

import Lux, Random, LinearAlgebra

@doc raw"""
This checks if the Adam cache was set up in the right way
"""
function check_adam_cache(C::AbstractCache{T}, tol= T(10) * eps(T)) where T 
    @test typeof(C) <: AdamCache 
    @test propertynames(C) == (:B₁, :B₂)
    @test typeof(C.B₁) <: StiefelLieAlgHorMatrix
    @test typeof(C.B₂) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(C.B₁) < tol
    @test LinearAlgebra.norm(C.B₂) < tol
end 
check_adam_cache(B::NamedTuple) = apply_toNT(check_adam_cache, B)

@doc raw"""
This checks if the momentum cache was set up in the right way
"""
function check_momentum_cache(C::AbstractCache{T}, tol= T(10) * eps(T)) where T 
    @test typeof(C) <: MomentumCache 
    @test propertynames(C) == (:B,)
    @test typeof(C.B) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(C.B) < tol
end
check_momentum_cache(B::NamedTuple) = apply_toNT(check_momentum_cache, B)

@doc raw"""
This checks if the gradient cache was set up in the right way
"""
function check_gradient_cache(C::AbstractCache{T}) where T 
    @test typeof(C) <: GradientCache 
    @test propertynames(C) == ()
end
check_gradient_cache(B::NamedTuple) = apply_toNT(check_gradient_cache, B)

@doc raw"""
This checks if all the caches are set up in the right way for the `MultiHeadAttention` layer with Stiefel weights.

TODO:
- [ ] `BFGSOptimizer` !!
"""
function test_cache_setups_for_optimizer_for_multihead_attention_layer(T::Type, dim::Int, n_heads::Int)
    @assert dim % n_heads == 0
    model = MultiHeadAttention(dim, n_heads, Stiefel=true)
    ps = initialparameters(CPU(), T, model)

    o₁ = Optimizer(AdamOptimizer(), ps)
    o₂ = Optimizer(MomentumOptimizer(), ps)
    o₃ = Optimizer(GradientOptimizer(), ps)

    check_adam_cache(o₁.cache)
    check_momentum_cache(o₂.cache)
    check_gradient_cache(o₃.cache)
end

test_cache_setups_for_optimizer_for_multihead_attention_layer(Float32, 64, 8)