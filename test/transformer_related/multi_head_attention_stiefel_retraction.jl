"""
This is a test for that checks if the retractions (geodesic and Cayley for now) map from StiefelLieAlgHorMatrix to StiefelManifold when used with MultiHeadAttention.
"""

import Random, Test, Lux, LinearAlgebra, KernelAbstractions

using GeometricMachineLearning, Test
using GeometricMachineLearning: geodesic
using GeometricMachineLearning: cayley

dim = 64
n_heads = 8
Dₕ = dim÷8
tol = eps(Float32)
T = Float32
backend = KernelAbstractions.CPU()

model = MultiHeadAttention(dim, n_heads, Stiefel=true)

ps = initialparameters(backend, T, model)

cache = init_optimizer_cache(MomentumOptimizer(), ps)

E = StiefelProjection(dim, Dₕ, T)
function check_retraction_geodesic(A::AbstractMatrix) 
    A_retracted = geodesic(A)
    @test typeof(A_retracted) <: StiefelManifold
    @test LinearAlgebra.norm(A_retracted - E) < tol
end
check_retraction_geodesic(cache::NamedTuple) = apply_toNT(check_retraction_geodesic, cache)
check_retraction_geodesic(B::MomentumCache) = check_retraction_geodesic(B.B)

check_retraction_geodesic(cache)

E = StiefelProjection(dim, Dₕ)
function check_retraction_cayley(A::AbstractMatrix) 
    A_retracted = cayley(A)
    @test typeof(A_retracted) <: StiefelManifold
    @test LinearAlgebra.norm(A_retracted - E) < tol
end
check_retraction_cayley(cache::NamedTuple) = apply_toNT(check_retraction_cayley, cache)
check_retraction_cayley(B::MomentumCache) = check_retraction_cayley(B.B)

check_retraction_cayley(cache)
