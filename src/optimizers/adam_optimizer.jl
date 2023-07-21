"""
Define the Adam Optimizer (no riemannian version yet!)
Algorithm and suggested defaults are taken from (Goodfellow et al., 2016, page 301), except for δ because single precision is used!
"""

struct AdamOptimizer{T<:Real} <: OptimizerMethod
    η::T
    ρ₁::T
    ρ₂::T
    δ::T

    AdamOptimizer(η = Float32(1e-3), ρ₁ = Float32(0.9), ρ₂ = Float32(0.99), δ = 3f-7) = new{typeof(η)}(η, ρ₁, ρ₂, δ)
end

function update!(o::Optimizer{<:AdamOptimizer{T}}, C::AdamCache, B::AbstractVecOrMat) where T
    add!(C.B₁, ((o.method.ρ₁ - o.method.ρ₁^o.step)/(T(1.) - o.method.ρ₁^o.step))*C.B₁, ((T(1.) - o.method.ρ₁)/(T(1.) - o.method.ρ₁^o.step))*B)
    add!(C.B₂, ((o.method.ρ₂ - o.method.ρ₂^o.step)/(T(1.) - o.method.ρ₂^o.step))*C.B₂, ((T(1.) - o.method.ρ₂)/(T(1.) - o.method.ρ₂^o.step))*⊙²(B))
    mul!(B, -o.method.η, /ᵉˡᵉ(C.B₁, scalar_add(racᵉˡᵉ(C.B₂), o.method.δ)))
end

#fallbacks: 
⊙²(A::AbstractVecOrMat) = A.^2
racᵉˡᵉ(A::AbstractVecOrMat) = sqrt.(A)
/ᵉˡᵉ(A::AbstractVecOrMat, B::AbstractVecOrMat) = A./B
scalar_add(A::AbstractVecOrMat, δ::Real) = A .+ δ

init_optimizer_cache(opt::AdamOptimizer, x) = setup_adam_cache(x)
