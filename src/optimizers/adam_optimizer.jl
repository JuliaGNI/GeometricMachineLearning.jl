"""
Define the Adam Optimizer (no riemannian version yet!)
Algorithm and suggested defaults are taken from (Goodfellow et al., 2016, page 301).
"""
struct AdamOptimizer{T} <: AbstractOptimizer
    η::T
    ρ₁::T
    ρ₂::T
    δ::T
    t::Int
    AdamOptimizer(η = 1e-3, ρ₁ = 0.9, ρ₂ = 0.99, δ = 1e-8) = new{typeof(η)}(η, ρ₁, ρ₂, δ, 0)
end

#update for single layer
function update!(o::AdamOptimizer, C::AdamLayerCache, B::NamedTuple)
    #o.t += 1
    for key in keys(B)
        C.B₁[key] = (o.ρ₁ - o.ρ₁^t)/(1 - ρ₁^t)*C.B₁[key] + (1 - ρ₁)/(1 - ρ₁^t)*B[key]
        C.B₂[key] = (o.ρ₂ - o.ρ₂^t)/(1 - ρ₂^t)*C.B₂[key] + (1 - ρ₂)/(1 - ρ₂^t)*⊙²(B[key])
        B[key] = (-o.η)*(/ᵉˡᵉ(C.B₁[key], scalar_add(√ᵉˡᵉ(C.B₂[key]), o.δ)))
    end
    B
end