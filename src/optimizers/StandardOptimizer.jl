"""
Define the Standard optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
struct StandardOptimizer{T} <: AbstractOptimizer
    η::T
    StandardOptimizer(η = 1e-2) = new{typeof(η)}(η)
end

init(o::StandardOptimizer, x) = nothing


function update_layer!(o::StandardOptimizer, state, ::Lux.AbstractExplicitLayer, x, dx)
    for obj in keys(x)
        x[obj] .-= o.η * dx[obj]
    end 
end

function update_layer!(o::StandardOptimizer, state, l::SymplecticStiefelLayer, x, dx)
    Manifolds.retract_caley!(l.manifold, x.weight, x.weight, -o.η * dx.weight)
end

function apply!(o::StandardOptimizer, state, model, x, dx)
    for layer in keys(model)
        update_layer!(o, state, model[layer], x[layer], dx[layer])
    end
end

apply!(o::StandardOptimizer, state, model::Chain, x, dx) = apply!(o, state, model.layers, x, dx)
