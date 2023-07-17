"""
Define the Gradient optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
struct GradientOptimizer{T<:Real} <: OptimizerMethod
    η::T
    t::Integer
    GradientOptimizer(η = 1e-2) = new{typeof(η)}(η,0)
end

function update!(o::GradientOptimizer, ::GradientCache, B::AbstractMatrix)
    rmul!(B, -o.η)
end

init_optimizer_cache(opt::GradientOptimizer, x) = setup_gradient_cache(x)
