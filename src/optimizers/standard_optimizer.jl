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

#TODO: Put this & horizontal_lift into a separate file 
#Riemannian Gradient: ∇f(U)UᵀU+JU(∇f(U))^TJU
function riemannian_gradient(U::AbstractMatrix, e_grad::AbstractMatrix, J::AbstractMatrix)
    (weight = e_grad * U' * U + J * U * e_grad' * J * U,)
end
