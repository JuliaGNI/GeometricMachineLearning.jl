"""
TODO: change this test so that it works with the new interface/framework!
"""

import Random, Test, Lux, LinearAlgebra

using GeometricMachineLearning, Test

dim = 64
n_heads = 8
Dₕ = dim÷8
tol = eps(Float32)

model = MultiHeadAttention(dim, n_heads, Stiefel=true)
#setting up "the gradients"
psᵧ, stᵧ = Lux.setup(TrivialInitRNG(), model)

ps₂ = retraction(model, psᵧ)

E = StiefelProjection(dim, Dₕ)
function check_retraction(A::AbstractMatrix) 
    @test typeof(A) <: StiefelManifold
    @test LinearAlgebra.norm(A - E) < tol
end
check_retraction(ps₂::NamedTuple) = apply_toNT(ps₂, check_retraction)

check_retraction(ps₂)

model = MultiHeadAttention(dim, n_heads, Stiefel=true, Retraction=Cayley())
psᵧ, stᵧ = Lux.setup(TrivialInitRNG(), model)

ps₂ = retraction(model, psᵧ)
check_retraction(ps₂)