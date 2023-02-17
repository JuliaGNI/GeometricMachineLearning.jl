"""
Define the Standard optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
struct StandardOptimizer{T} <: AbstractOptimizer
    η::T
    StandardOptimizer(η = 1e-2) = new{typeof(η)}(η)
end

setup(::StandardOptimizer, x) = NamedTuple()

function update_layer!(o::StandardOptimizer, state, layer, x, dx)
    update_layer!(layer, x, dx, -o.η)
end

function update_layer!(o::StandardOptimizer, state, layer::ManifoldLayer, x, dx)
    update_layer!(layer, x, riemannian_gradient(dx.weight, x.weight, layer.sympl_out), -o.η)
end

#Riemannian Gradient: ∇f(U)UᵀU+JU(∇f(U))^TJU
function riemannian_gradient(e_grad, U, J)
    e_grad * U' * U + J * U * e_grad' * J * U
end
