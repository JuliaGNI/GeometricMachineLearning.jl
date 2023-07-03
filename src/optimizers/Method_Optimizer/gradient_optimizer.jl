"""
Define the Gradient optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct GradientOptimizer{T<:Real} <: AbstractMethodOptimiser
    η::T
    t::Integer
    GradientOptimizer(η = 1e-2) = new{typeof(η)}(η,0)
end

function update!(o::GradientOptimizer, ::GradientCache, B::AbstractMatrix)
    rmul!(B, -o.η)
end

init_optimizer_cache(d::Lux.AbstractExplicitLayer, ::GradientOptimizer) = setup_gradient_cache(d)

