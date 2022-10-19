"""
Define the Standard optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""

struct StandardOptimizer{T} <: AbstractOptimizer
    η::T
end
StandardOptimizer(η = 1f-2) = StandardOptimizer{typeof(η)}(η)

function apply!(o::StandardOptimizer, x, dx, st::NamedTuple)
    layer_names = keys(st)
    i = 0
    for layer in layer_names
        i += 1
        if isempty(st[layer])
            for obj in keys(ps[layer])
                x[layer][obj] .-= o.η * dx[layer][obj]
            end 
        else
            Manifolds.retract_caley!(st[layer].Manifold, 
            x[layer].weight, x[layer].weight, -o.η * dx[layer].weight)
        end
    end
end