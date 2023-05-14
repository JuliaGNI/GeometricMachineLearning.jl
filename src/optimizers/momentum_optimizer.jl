"""
Define the Momentum optimizer, i.e. 
V ← α*V - ∇f(W)
W ← W + η*V
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct MomentumOptimizer{T<:Real} <: AbstractOptimizer
    η::T
    α::T
    t::Int
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α, 0)
end

#update for weights
function update!(o::MomentumOptimizer, C::MomentumCache, B::AbstractMatrix)
    C.B = o.α*C.B + B
    B = -o.η*C.B
end

init_optimizer_cache(d::Lux.AbstractExplicitLayer, ::MomentumOptimizer) = setup_momentum_cache(d)