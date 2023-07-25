"""
Define the Gradient optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
struct GradientOptimizer{T<:Real} <: OptimizerMethod
    η::T
    GradientOptimizer(η = 1e-2) = new{typeof(η)}(η)
end

function update!(o::Optimizer{<:GradientOptimizer}, ::GradientCache, B::AbstractVecOrMat)
    rmul!(B, -o.method.η)
end

init_optimizer_cache(opt::GradientOptimizer, x) = setup_gradient_cache(x)
