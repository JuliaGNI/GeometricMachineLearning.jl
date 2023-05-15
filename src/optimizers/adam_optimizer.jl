"""
Define the Adam Optimizer (no riemannian version yet!)
Algorithm and suggested defaults are taken from (Goodfellow et al., 2016, page 301).
"""
mutable struct AdamOptimizer{T<:Real} <: AbstractOptimizer
    η::T
    ρ₁::T
    ρ₂::T
    δ::T
    t::Int
    AdamOptimizer(η = Float32(1e-3), ρ₁ = Float32(0.9), ρ₂ = Float32(0.99), δ = Float32(1e-8)) = new{typeof(η)}(η, ρ₁, ρ₂, δ, 0)
end

function update!(o::AdamOptimizer, C::AdamCache, B::AbstractVecOrMat)
    add!(C.B₁, (o.ρ₁ - o.ρ₁^o.t)/(1 - o.ρ₁^o.t)*C.B₁, (1 - o.ρ₁)/(1 - o.ρ₁^o.t)*B)
    add!(C.B₂, (o.ρ₂ - o.ρ₂^o.t)/(1 - o.ρ₂^o.t)*C.B₂, (1 - o.ρ₂)/(1 - o.ρ₂^o.t)*⊙²(B))
    mul!(B, -o.η, /ᵉˡᵉ(C.B₁, scalar_add(√ᵉˡᵉ(C.B₂), o.δ)))
end

#fallbacks: 
⊙²(A::AbstractMatrix) = A.^2
√ᵉˡᵉ(A::AbstractMatrix) = sqrt.(A)
/ᵉˡᵉ(A::AbstractMatrix, B::AbstractMatrix) = A./B
scalar_add(A::AbstractMatrix, δ::Real) = A .+ δ

init_optimizer_cache(d::Lux.AbstractExplicitLayer, ::AdamOptimizer) = setup_adam_cache(d)
