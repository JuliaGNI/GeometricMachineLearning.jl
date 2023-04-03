"""
Define the Standard optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct StandardOptimizer{T} <: AbstractOptimizer
    η::T
    StandardOptimizer(η = 1e-2) = new{typeof(η)}(η)
end

setup(::StandardOptimizer, x) = NamedTuple()

function update_layer!(o::StandardOptimizer, state, layer::Lux.AbstractExplicitLayer,
                       x::NamedTuple, dx::NamedTuple)
    update_layer!(layer, x, dx, -o.η)
end

function update_layer!(o::StandardOptimizer, state, layer::ManifoldLayer, x::NamedTuple,
                       dx::NamedTuple)
    update_layer!(layer, x, riemannian_gradient(x.weight, dx.weight, layer.sympl_out), -o.η)
end
