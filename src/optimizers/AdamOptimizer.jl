"""
Define the Adam Optimizer (no riemannian version yet!)
Algorithm and suggested defaults are taken from (Goodfellow et al., 2016, page 301).
"""
struct AdamOptimizer{T} <: AbstractOptimizer
    η::T
    ρ₁::T
    ρ₂::T
    δ::T
    AdamOptimizer(η = 1e-3, ρ₁ = 0.9, ρ₂ = 0.99, δ = 1e-8) = new{typeof(η)}(η, ρ₁, ρ₂, δ)
end

init(o::AdamOptimizer, x) = nothing

#normally it should be "obj in keys(x)"; but this isn't possible because of HNN hack
function update_layer!(o::AdamOptimizer, state, ::Lux.AbstractExplicitLayer, x, dx)
    for obj in 1:length(x)
        state[1][obj] .= o.ρ₁ * state[1][obj] + (1.0 - o.ρ₁) * dx[obj]
        state[2][obj] .= o.ρ₂ * state[2][obj] + (1.0 - o.ρ₂) * dx[obj] .^ 2
        x[obj] .-= o.η * (1.0 - o.ρ₁^state[3]) * state[1][obj] ./
                   (sqrt.((1.0 - o.ρ₂^state[3]) * state[2][obj]) .+ o.δ)
    end
end

function update_layer!(o::AdamOptimizer, state, model::Lux.Chain, x, dx)
    for i in length(model)
        update_layer!(o, (state[1][i], state[2][i], state[3]), model[i], x[i], dx[i])
    end
    print(state[3])
    state[3] += 1
end

function apply!(o::AdamOptimizer, state, model::Lux.Chain, x, dx)
    update_layer!(o, state, model, x, dx)
end

#initialize Adam
function init_adam!(::Lux.AbstractExplicitLayer, x::NamedTuple)
    for obj in x
        obj .= zeros(size(obj))
    end
end

function init_adam!(model::Lux.Chain, x::NamedTuple)
    for index in 1:length(model)
        init_adam!(model[index], x[index])
    end
end

function init_adam(model::Lux.AbstractExplicitLayer)
    ps, st = Lux.setup(Random.default_rng(), model)
    init_adam!(model, ps)
    (ps, deepcopy(ps), 1)
end