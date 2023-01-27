"""
Define the Standard optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
struct StandardOptimizer{T} <: AbstractOptimizer
    η::T
    StandardOptimizer(η = 1e-2) = new{typeof(η)}(η)
end

setup(o::StandardOptimizer, x) = NamedTuple()

function update_layer!(o::StandardOptimizer, state, ::Lux.AbstractExplicitLayer, x, dx)
    for obj in keys(x)
        x[obj] .-= o.η * dx[obj]
    end
end

#Riemannian Gradient: ∇f(U)UᵀU+JU(∇f(U))^TJU
function r_grad(e_grad, U, J)
    e_grad * U' * U + J * U * e_grad' * J * U
end

function update_layer!(o::StandardOptimizer, state, l::SymplecticStiefelLayer, x, dx)
    Manifolds.retract_caley!(l.manifold, x.weight, copy(x.weight),
                             -o.η * r_grad(dx.weight, x.weight, l.sympl_out))
end

function apply!(o::StandardOptimizer, state, model::Chain, x, dx)
    for i in 1:length(model)
        update_layer!(o, state, model[i], x[i], dx[i])
    end
end
