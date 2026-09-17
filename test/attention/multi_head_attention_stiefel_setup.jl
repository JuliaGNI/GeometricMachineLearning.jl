using GeometricMachineLearning, Test
# see the note in `multi_head_attention_stiefel_retraction.jl`
using GeometricOptimizers: MomentumCache
import Random, Test, LinearAlgebra, KernelAbstractions
# The cache and the state hold a `NetworkParameters` per layer: a layer is wrapped at the
# `GeometricOptimizers` boundary, and the wrap shares the leaf arrays. `mapparameters` and not `map`,
# because it recurses on the branches and so reaches the leaves whichever shape it is handed.
#
# Each helper below has two methods, because two shapes arrive: a wrapped layer is a
# `NetworkParameters` and the tree above it is a plain `NamedTuple`.
using NeuralNetworkParameters: NetworkParameters, mapparameters

Random.seed!(1234)

@doc raw"""
This checks for an arbitrary matrix ``A\in\mathbb{R}^{N\times{}n}`` if ``A\in{}St(n,N)``.
"""
function check_setup(A::AbstractMatrix{T}, tol = T(10)*eps(T)) where {T}
    @test typeof(A) <: StiefelManifold
    @test check(A) < tol
end
check_setup(ps::NetworkParameters) = mapparameters(check_setup, ps)
check_setup(ps::NamedTuple) = mapparameters(check_setup, ps)
check_setup(ps::NetworkParameters) = check_setup(GeometricMachineLearning.params(ps))

@doc raw"""
This checks for an arbitrary matrix ``B\in\mathbb{R}^{N\times{}N}`` if ``B\in\mathfrak{g}^\mathrm{hor}``.
"""
function check_grad_setup(B::AbstractMatrix{T}, tol = T(10)*eps(T)) where {T}
    @test typeof(B) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(B) < tol
end
check_grad_setup(gx::NetworkParameters) = mapparameters(check_grad_setup, gx)
check_grad_setup(gx::NamedTuple) = mapparameters(check_grad_setup, gx)
check_grad_setup(B::MomentumCache) = check_grad_setup(B.δ)

@doc raw"""
Check if `initialparameters` and `init_optimizer_cache` do the right thing for `MultiHeadAttentionLayer`.
"""
function check_multi_head_attention_stiefel_setup(T::Type, N::Int, n::Int)
    model = Chain(MultiHeadAttention(N, n, Stiefel = true))
    ps = GeometricMachineLearning.params(NeuralNetwork(model, KernelAbstractions.CPU(), T))

    check_setup(ps)

    gx = Optimizer(MomentumMethod(), ps).cache
    check_grad_setup(gx)
end

check_multi_head_attention_stiefel_setup(Float32, 64, 8)
