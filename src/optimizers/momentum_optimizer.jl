"""
Define the Momentum optimizer, i.e. 
V ← α*V - ∇f(W)
W ← W + η*V
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct MomentumOptimizer{T} <: AbstractOptimizer
    η::T
    α::T
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α)
end

#TODO: put this and riemannian_gradient into a separate file (functions associated with the Stiefel Manifold! - probably a good idea to generalize this to other manifolds!)
function horizontal_lift(U::AbstractMatrix, Δ::AbstractMatrix, J::AbstractMatrix)
    U_inv = (U' * U)^-1
    Δ * U_inv * U' + J * U * U_inv * Δ' * (I - J' * U * U_inv * U' * J) * J
end

#sympl_out saves the ``big'' Js. 
#dx is the EUCLIDEAN gradient!
function update_layer!(o::MomentumOptimizer, state::MomentumOptimizerLayerCache,
                       layer::SymplecticStiefelLayer, x::NamedTuple, dx::NamedTuple)
    state.momentum.weight .= (o.α *
                              horizontal_lift(state.prev_step.weight, state.momentum.weight,
                                              layer.sympl_out) -
                              horizontal_lift(x.weight,
                                              riemannian_gradient(x.weight, dx.weight,
                                                                  layer.sympl_out).weight,
                                              layer.sympl_out)) * x.weight
    state.prev_step.weight .= copy(x.weight)
    update_layer!(layer, x, state.momentum, o.η)
end

function update_layer!(o::MomentumOptimizer, state::MomentumOptimizerLayerCache,
                       layer::Lux.AbstractExplicitLayer, x::NamedTuple, dx::NamedTuple)
    for obj in keys(x)
        state.momentum[obj] .= o.α * state.momentum[obj] - dx[obj]
    end
    update_layer!(layer, x, state.momentum, o.η)
end

#=
function apply!(o::MomentumOptimizer, state::MomentumOptimizerCache, model::Lux.Chain, x::NamedTuple,
                dx::NamedTuple)
    for i in 1:length(model)
        layer_name = Symbol("layer_$i")
        update_layer!(o, state.state[layer_name], model[i], x[layer_name], dx[layer_name])
    end
end
=#