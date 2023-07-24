"""
Define the Momentum optimizer, i.e. 
V ← α*V - ∇f(W)
W ← W + η*V
Or the riemannian manifold equivalent, if applicable.
"""
struct MomentumOptimizer{T<:Real} <: OptimizerMethod
    η::T
    α::T
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α)
end

#update for weights
function update!(o::MomentumOptimizer, C::MomentumCache, B::AbstractVecOrMat)
    add!(C.B, o.α*C.B, B)
    mul!(B, -o.η, C.B)
end

init_optimizer_cache(opt::MomentumOptimizer, x) = setup_momentum_cache(x)
