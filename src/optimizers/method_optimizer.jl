"""
This file defines all the optimization methods including 
- the structure
- the update function (for a single layer)
"""

abstract type AbstractMethodOptimiser end


"""
Define the Standard optimizer, i.e. W ← W - η*∇f(W)
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct GradientOptimizer{T} <: AbstractMethodOptimiser
    η::T
    t::Integer
    GradientOptimizer(η = 1e-2) = new{typeof(η)}(η,0)
end

function update!(o::GradientOptimizer, ::StandardLayerCache, B::NamedTuple)
    #o.t += 1
    for key in keys(B)
        B[key] .= -o.η*B[key]
    end
    B
end

"""
Define the Momentum optimizer, i.e. 
V ← α*V - ∇f(W)
W ← W + η*V
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct MomentumOptimizer{T} <: AbstractMethodOptimiser
    η::T
    α::T
    t::Int
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α, 0)
end

#update for single layer
function update!(o::MomentumOptimizer, C::MomentumLayerCache, B::NamedTuple)
    #o.t += 1
    for key in keys(B)
        C.B[key] .= o.α*C.B[key] + B[key]
        B[key] .= -o.η*C.B[key]
    end 
    B
end


"""
Define the Adam Optimizer (no riemannian version yet!)
Algorithm and suggested defaults are taken from (Goodfellow et al., 2016, page 301).
"""
mutable struct AdamOptimizer{T} <: AbstractMethodOptimiser 
    η::T
    ρ₁::T
    ρ₂::T
    δ::T
    t::Int
    AdamOptimizer(η = 1e-3, ρ₁ = 0.9, ρ₂ = 0.99, δ = 1e-8) = new{typeof(η)}(η, ρ₁, ρ₂, δ, 0)
end

#update for single layer
function update!(o::AdamOptimizer, C::AdamLayerCache, B::NamedTuple)
    for key in keys(B)
        C.B₁[key] .= (o.ρ₁ - o.ρ₁^o.t)/(1 - o.ρ₁^o.t)*C.B₁[key] .+ (1 - o.ρ₁)/(1 - o.ρ₁^o.t)*B[key]
        C.B₂[key] .= (o.ρ₂ - o.ρ₂^o.t)/(1 - o.ρ₂^o.t)*C.B₂[key] .+ (1 - o.ρ₂)/(1 - o.ρ₂^o.t)*⊙²(B[key])
        B[key] .= (-o.η)*(/ᵉˡᵉ(C.B₁[key], scalar_add(√ᵉˡᵉ(C.B₂[key]), o.δ)))
    end
    B
end

#fallbacks: 
⊙²(A::AbstractMatrix) = A.^2
√ᵉˡᵉ(A::AbstractMatrix) = sqrt.(A)
/ᵉˡᵉ(A::AbstractMatrix, B::AbstractMatrix) = A./B
scalar_add(A::AbstractMatrix, δ::Real) = A .+ δ
#⊙²(a::Real) = a^2
#√ᵉˡᵉ(a::Real) = sqrt(a)
#/ᵉˡᵉ(a::Real, b::Real) = a/b
#scalar_add(a::Real, δ::Real) = a + δ