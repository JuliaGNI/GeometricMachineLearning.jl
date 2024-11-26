@doc raw"""
    AdamOptimizer(η, ρ₁, ρ₂, δ)

Make an instance of the Adam Optimizer.

Here the cache consists of first and second moments that are updated as 

```math
B_1 \gets ((\rho_1 - \rho_1^t)/(1 - \rho_1^t))\cdot{}B_1 + (1 - \rho_1)/(1 - \rho_1^t)\cdot{}\nabla{}L,
```
and

```math
B_2 \gets ((\rho_2 - \rho_1^t)/(1 - \rho_2^t))\cdot{}B_2 + (1 - \rho_2)/(1 - \rho_2^t)\cdot\nabla{}L\odot\nabla{}L.
```
The final velocity is computed as:

```math
\mathrm{velocity} \gets -\eta{}B_1/\sqrt{B_2 + \delta}.
```

# Implementation 

The *velocity* is stored in the input to save memory:

```julia
mul!(B, -o.method.η, /ᵉˡᵉ(C.B₁, scalar_add(racᵉˡᵉ(C.B₂), o.method.δ)))
```
where `B` is the input to the [`update!`] function.

The algorithm and suggested defaults are taken from [goodfellow2016deep; page 301](@cite).
"""
struct AdamOptimizer{T<:Real} <: OptimizerMethod{T}
    η::T
    ρ₁::T
    ρ₂::T
    δ::T

    AdamOptimizer(η = 1f-3, ρ₁ = 9f-1, ρ₂ = 9.9f-1, δ = 3f-7; T=typeof(η)) = new{T}(T(η), T(ρ₁), T(ρ₂), T(δ))
end

function AdamOptimizer(T::Type)
    AdamOptimizer(T(1f-3))
end

function update!(o::Optimizer{<:AdamOptimizer{T}}, C::AdamCache, B::AbstractArray) where T
    add!(C.B₁, ((o.method.ρ₁ - o.method.ρ₁^o.step)/(T(1.) - o.method.ρ₁^o.step))*C.B₁, ((T(1.) - o.method.ρ₁)/(T(1.) - o.method.ρ₁^o.step))*B)
    add!(C.B₂, ((o.method.ρ₂ - o.method.ρ₂^o.step)/(T(1.) - o.method.ρ₂^o.step))*C.B₂, ((T(1.) - o.method.ρ₂)/(T(1.) - o.method.ρ₂^o.step))*⊙²(B))
    mul!(B, -o.method.η, /ᵉˡᵉ(C.B₁, scalar_add(racᵉˡᵉ(C.B₂), o.method.δ)))
end

# defaults: 
⊙²(A::AbstractVecOrMat) = A.^2
racᵉˡᵉ(A::AbstractVecOrMat) = sqrt.(A)
/ᵉˡᵉ(A::AbstractVecOrMat, B::AbstractVecOrMat) = A./B
scalar_add(A::AbstractVecOrMat, δ::Real) = A .+ δ