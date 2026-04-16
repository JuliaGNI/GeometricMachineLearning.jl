using GeometricMachineLearning, Test
using GeometricMachineLearning: init_optimizer_cache
import Random, Test, LinearAlgebra, KernelAbstractions

Random.seed!(1234)

@doc raw"""
This checks for an arbitrary matrix ``A\in\mathbb{R}^{N\times{}n}`` if ``A\in{}St(n,N)``.
"""
function check_setup(A::AbstractMatrix{T}, tol=T(10)*eps(T)) where T
    @test typeof(A) <: StiefelManifold
    @test check(A) < tol
end
check_setup(ps::NamedTuple) = apply_toNT(check_setup, ps)
check_setup(ps::NeuralNetworkParameters) = check_setup(GeometricMachineLearning.params(ps))

@doc raw"""
This checks for an arbitrary matrix ``B\in\mathbb{R}^{N\times{}N}`` if ``B\in\mathfrak{g}^\mathrm{hor}``.
"""
function check_grad_setup(B::AbstractMatrix{T}, tol=T(10)*eps(T)) where T
    @test typeof(B) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(B) < tol
end
check_grad_setup(gx::NamedTuple) = apply_toNT(check_grad_setup, gx)
check_grad_setup(B::MomentumCache) = check_grad_setup(B.B)

@doc raw"""
Check if `initialparameters` and `init_optimizer_cache` do the right thing for `MultiHeadAttentionLayer`.
"""
function check_multi_head_attention_stiefel_setup(T::Type, N::Int, n::Int)
    model = Chain(MultiHeadAttention(N, n, Stiefel=true))
    ps = GeometricMachineLearning.params(NeuralNetwork(model, KernelAbstractions.CPU(), T))

    check_setup(ps)

    gx = init_optimizer_cache(MomentumOptimizer(), ps)
    check_grad_setup(gx)
end

check_multi_head_attention_stiefel_setup(Float32, 64, 8)