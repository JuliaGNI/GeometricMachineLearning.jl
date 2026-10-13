using GeometricMachineLearning, Test
using GeometricMachineLearning: geodesic
using GeometricMachineLearning: cayley
# The optimizer caches are internal to GeometricOptimizers -- they are `solver_step!` scratch --
# so they are named qualified rather than through a re-export.
using GeometricOptimizers: MomentumCache
import Random, Test, LinearAlgebra, KernelAbstractions
# One cache covers the whole network, and its fields hold a `NetworkParameters` of the network's
# shape. `mapparameters` and not `map`, because it recurses on the branches and so reaches the leaves.
#
# Each helper below has a method for a `NetworkParameters` and one for a plain `NamedTuple`, so that
# it reaches the leaves whichever of the two it is handed.
using NeuralNetworkParameters: NetworkParameters, mapparameters

Random.seed!(1234)

@doc raw"""
This function computes the geodesic retraction of an element of `StiefelLieAlgHorMatrix` and then checks if the resulting element is `StiefelProjection`.
"""
function check_retraction_geodesic(A::AbstractMatrix{T}, tol = eps(T)) where {T}
    A_retracted = geodesic(A)
    @test typeof(A_retracted) <: StiefelManifold
    @test LinearAlgebra.norm(A_retracted - StiefelProjection(A_retracted)) < tol
end
function check_retraction_geodesic(cache::NetworkParameters)
    mapparameters(check_retraction_geodesic, cache)
end
function check_retraction_geodesic(cache::NamedTuple)
    mapparameters(check_retraction_geodesic, cache)
end
check_retraction_geodesic(B::MomentumCache) = check_retraction_geodesic(B.δ)

@doc raw"""
This function computes the cayley retraction of an element of `StiefelLieAlgHorMatrix` and then checks if the resulting element is `StiefelProjection`.
"""
function check_retraction_cayley(A::AbstractMatrix{T}, tol = eps(T)) where {T}
    A_retracted = cayley(A)
    @test typeof(A_retracted) <: StiefelManifold
    @test LinearAlgebra.norm(A_retracted - StiefelProjection(A_retracted)) < tol
end
function check_retraction_cayley(cache::NetworkParameters)
    mapparameters(check_retraction_cayley, cache)
end
check_retraction_cayley(cache::NamedTuple) = mapparameters(check_retraction_cayley, cache)
check_retraction_cayley(B::MomentumCache) = check_retraction_cayley(B.δ)

@doc raw"""
This is a test for that checks if the retractions (geodesic and Cayley for now) map from `StiefelLieAlgHorMatrix` to `StiefelManifold` when used with `MultiHeadAttention`.
"""
function test_multi_head_attention_retraction(
        T::Type, dim, n_heads, tol = eps(T), backend = KernelAbstractions.CPU())
    model = Chain(MultiHeadAttention(dim, n_heads, Stiefel = true))

    ps = NeuralNetwork(model, backend, T).params
    cache = TrainingOptimizer(ps; algorithm = MomentumMethod()).cache
    @test cache isa MomentumCache

    check_retraction_geodesic(cache)

    check_retraction_cayley(cache)
end

test_multi_head_attention_retraction(Float32, 64, 8)
