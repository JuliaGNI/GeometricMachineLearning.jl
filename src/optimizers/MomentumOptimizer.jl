"""
Define the Momentum optimizer, i.e. 
V ← α*V - ∇f(W)
W ← W + η*V
Or the riemannian manifold equivalent, if applicable.
"""
struct MomentumOptimizer{T} <: AbstractOptimizer
    η::T
    α::T
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α)
end

init(o::MomentumOptimizer, x) = nothing

#SymplecticMatrix is included in Manifolds.jl -> this is probably not very efficient
function horizontal_lift(U, Δ, J)
    U_inv = (U' * U)^-1
    Δ * U_inv * U' + J * U * U_inv * Δ' * (I - J' * U * U_inv * U' * J) * J
end

#Riemannian Gradient: ∇f(U)UᵀU+JU(∇f(U))^TJU
#function r_grad(U, e_grad, J)
#   e_grad * U' * U + J * U * e_grad' * J * U
#end

#sympl_out saves the ``big'' Js. 
function update_layer!(o::MomentumOptimizer, state, l::SymplecticStiefelLayer, x, dx)
    state.weight .-= o.α *
                     horizontal_lift(x.weight, r_grad(x.weight, dx.weight, l.sympl_out),
                                     l.sympl_out)
    Manifolds.retract_caley!(l.manifold, x.weight, x.weight, o.η * state.weight * x.weight)
end

function update_layer!(o::MomentumOptimizer, state, ::Lux.AbstractExplicitLayer, x, dx)
    for obj in keys(x)
        state[obj] .-= o.α * dx[obj]
        x[obj] .+= o.η * dx[obj]
    end
end

function update_layer!(o::MomentumOptimizer, state, model::Lux.Chain, x, dx)
    for i in 1:length(model)
        update_layer!(o, state[i], model[i], x[i], dx[i])
    end
end

function apply!(o::MomentumOptimizer, state, model::Lux.Chain, x, dx)
    update_layer!(o, state, model, x, dx)
end

#initialize Adam
function init_momentum!(::Lux.AbstractExplicitLayer, x::NamedTuple)
    for obj in x
        obj .= zeros(size(obj))
    end
end

function init_momentum!(l::SymplecticStiefelLayer, x::NamedTuple)
    x.weight .= zeros(l.dim_N, l.dim_N)
end

function init_momentum!(model::Lux.Chain, x::NamedTuple)
    for index in 1:length(model)
        init_momentum!(model[index], x[index])
    end
end

function init_momentum(model::Lux.AbstractExplicitLayer)
    ps, st = Lux.setup(Random.default_rng(), model)
    init_momentum!(model, ps)
    return ps
end
