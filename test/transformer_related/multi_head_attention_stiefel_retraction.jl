import Random, Test, Lux, LinearAlgebra, KernelAbstractions

using GeometricMachineLearning, Test
using GeometricMachineLearning: geodesic
using GeometricMachineLearning: cayley
using GeometricMachineLearning: init_optimizer_cache

@doc raw"""
This function computes the geodesic retraction of an element of `StiefelLieAlgHorMatrix` and then checks if the resulting element is `StiefelProjection`.
"""
function check_retraction_geodesic(A::AbstractMatrix{T}, tol=eps(T)) where T
    A_retracted = geodesic(A)
    @test typeof(A_retracted) <: StiefelManifold
    @test LinearAlgebra.norm(A_retracted - StiefelProjection(A_retracted)) < tol
end
check_retraction_geodesic(cache::NamedTuple) = apply_toNT(check_retraction_geodesic, cache)
check_retraction_geodesic(B::MomentumCache) = check_retraction_geodesic(B.B)

@doc raw"""
This function computes the cayley retraction of an element of `StiefelLieAlgHorMatrix` and then checks if the resulting element is `StiefelProjection`.
"""
function check_retraction_cayley(A::AbstractMatrix{T}, tol=eps(T)) where T
    A_retracted = cayley(A)
    @test typeof(A_retracted) <: StiefelManifold
    @test LinearAlgebra.norm(A_retracted - StiefelProjection(A_retracted)) < tol
end
check_retraction_cayley(cache::NamedTuple) = apply_toNT(check_retraction_cayley, cache)
check_retraction_cayley(B::MomentumCache) = check_retraction_cayley(B.B)

@doc raw"""
This is a test for that checks if the retractions (geodesic and Cayley for now) map from `StiefelLieAlgHorMatrix` to `StiefelManifold` when used with `MultiHeadAttention`.
"""
function test_multi_head_attention_retraction(T::Type, dim, n_heads, tol=eps(T), backend=KernelAbstractions.CPU())
    model = MultiHeadAttention(dim, n_heads, Stiefel=true)

    ps = initialparameters(backend, T, model)
    cache = init_optimizer_cache(MomentumOptimizer(), ps)

    check_retraction_geodesic(cache)

    check_retraction_cayley(cache)
end

test_multi_head_attention_retraction(Float32, 64, 8)