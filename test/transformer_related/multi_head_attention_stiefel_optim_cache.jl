using GeometricMachineLearning, Test

import Lux, Random, LinearAlgebra

dim = 64
n_heads = 8
Dₕ = dim÷8
tol = eps(Float32)

model = Chain(MultiHeadAttention(dim, n_heads), MultiHeadAttention(dim, n_heads, Stiefel=true))
ps = initialparameters(CPU(), Float32, model)

o₁ = Optimizer(AdamOptimizer(), ps)
o₂ = Optimizer(MomentumOptimizer(), ps)
o₃ = Optimizer(GradientOptimizer(), ps)

function check_adam_cache(C::AbstractCache) 
    @test typeof(C) <: AdamCache 
    @test propertynames(C) == (:B₁, :B₂)
    @test typeof(C.B₁) <: StiefelLieAlgHorMatrix
    @test typeof(C.B₂) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(C.B₁) < tol
    @test LinearAlgebra.norm(C.B₂) < tol
end 
check_adam_cache(B::NamedTuple) = apply_toNT(check_adam_cache, B)

function check_momentum_cache(C::AbstractCache)
    @test typeof(C) <: MomentumCache 
    @test propertynames(C) == (:B,)
    @test typeof(C.B) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(C.B) < tol
end
check_momentum_cache(B::NamedTuple) = apply_toNT(check_momentum_cache, B)

function check_gradient_cache(C::AbstractCache)
    @test typeof(C) <: GradientCache 
    @test propertynames(C) == ()
end
check_gradient_cache(B::NamedTuple) = apply_toNT(check_gradient_cache, B)

check_adam_cache(o₁.cache[2])
check_momentum_cache(o₂.cache[2])
check_gradient_cache(o₃.cache[2])
