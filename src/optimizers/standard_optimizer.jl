"""
Define the Standard optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct StandardOptimizer{T<:Real} <: AbstractOptimizer
    η::T
    t::Integer
    StandardOptimizer(η = 1e-2) = new{typeof(η)}(η,0)
end

function update!(o::StandardOptimizer, ::StandardCache, B::AbstractMatrix)
    rmul!(B, -o.η)
end

init_optimizer_cache(d::Lux.AbstractExplicitLayer, ::StandardOptimizer) = setup_standard_cache(d)