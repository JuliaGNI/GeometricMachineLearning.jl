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

function setup(o::MomentumOptimizer, model::Lux.AbstractExplicitLayer, x, dx)
    #this stores the previous x-steps for manifold layers
    x_prev = []
    indices = init_prev!(model, x_prev, x)
    names = ntuple(i -> Symbol("layer_$(indices[i])"), length(indices))
    layers = NamedTuple{names}(x_prev)
    #make standard optimization step (frist iteration)
    o₂ = StandardOptimizer(o.η)
    apply!(o₂, nothing, model, x, dx)
    #merge everything
    fun1(a, b) = NamedTuple{(:momentum, :prev_step)}((a, b))
    NamedTuple(mergewith(fun1, Dict(pairs(dx)), Dict(pairs(layers))))
end

function init_prev!(::Lux.AbstractExplicitLayer, x_prev, x, indices, index)
end

function init_prev!(::SymplecticStiefelLayer, x_prev, x, indices, index)
    push!(x_prev, x)
    push!(indices, index)
end

function init_prev!(model::Lux.Chain, x_prev, x)
    indices = []
    for index in 1:length(model)
        init_prev!(model[index], x_prev, x[index], indices, index)
    end
    indices
end

#SymplecticMatrix is included in Manifolds.jl -> this is probably not very efficient
function horizontal_lift(U, Δ, J)
    U_inv = (U' * U)^-1
    Δ * U_inv * U' + J * U * U_inv * Δ' * (I - J' * U * U_inv * U' * J) * J
end

#sympl_out saves the ``big'' Js. 
function update_layer!(o::MomentumOptimizer, state, l::SymplecticStiefelLayer, x, dx)
    state.momentum.weight .= (o.α *
                              horizontal_lift(state.prev_step.weight, state.momentum.weight,
                                              l.sympl_out) -
                              horizontal_lift(x.weight,
                                              r_grad(dx.weight, x.weight, l.sympl_out),
                                              l.sympl_out)) * x.weight
    state.prev_step.weight .= copy(x.weight)
    Manifolds.retract_caley!(l.manifold, x.weight, copy(x.weight),
                             o.η * state.momentum.weight)
end

function update_layer!(o::MomentumOptimizer, state, ::Lux.AbstractExplicitLayer, x, dx)
    for obj in keys(x)
        state[obj] .= o.α * state[obj] - dx[obj]
        x[obj] .+= o.η * state[obj]
    end
end

function update_layer!(o::MomentumOptimizer, state, model::Lux.Chain, x, dx)
    for i in 1:length(model)
        layer_name = Symbol("layer_$i")
        update_layer!(o, state[layer_name], model[i], x[layer_name], dx[layer_name])
    end
end

function apply!(o::MomentumOptimizer, state, model::Lux.Chain, x, dx)
    update_layer!(o, state, model, x, dx)
end
