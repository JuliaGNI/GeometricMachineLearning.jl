@doc raw"""
Defines the Adam Optimizer with weight decay.

### Constructors
The default constructor takes as input: 
- `n_epochs::Int`
- `η₁`: the learning rate at the start 
- `η₂`: the learning rate at the end 
- `ρ₁`: the decay parameter for the first moment 
- `ρ₂`: the decay parameter for the second moment
- `δ`: the safety parameter 
- `T` (keyword argument): the type. 

The second constructor is called with: 
- `n_epochs::Int`
- `T`
... the rest are keyword arguments
"""
struct AdamOptimizerWithDecay{T<:Real} <: OptimizerMethod
    η₁::T
    η₂::T
    ρ₁::T
    ρ₂::T
    δ::T
    γ::T
    n_epochs::Int

    function AdamOptimizerWithDecay(n_epochs::Int, η₁=1f-2, η₂=1f-6, ρ₁=9f-1, ρ₂=9.9f-1, δ=1f-8; T=typeof(η₁))
        γ = exp(log(η₂ / η₁) / n_epochs)
        new{T}(T(η₁), T(η₂), T(ρ₁), T(ρ₂), T(δ), T(γ), n_epochs)
    end
end

function AdamOptimizerWithDecay(n_epochs::Int, T::Type; η₁=1f-2, η₂=1f-6, ρ₁=9f-1, ρ₂=9.9f-1, δ=1f-8)
    AdamOptimizerWithDecay(n_epochs, T(η₁), T(η₂), T(ρ₁), T(ρ₂), T(δ))
end

function update!(o::Optimizer{<:AdamOptimizerWithDecay{T}}, C::AdamCache, B::AbstractArray) where T
    η = o.method.γ ^ o.step * o.method.η₁
    add!(C.B₁, ((o.method.ρ₁ - o.method.ρ₁^o.step) / (T(1.) - o.method.ρ₁^o.step)) * C.B₁, ((T(1.) - o.method.ρ₁) / (T(1.) - o.method.ρ₁ ^ o.step)) * B)
    add!(C.B₂, ((o.method.ρ₂ - o.method.ρ₂^o.step) / (T(1.) - o.method.ρ₂^o.step)) * C.B₂, ((T(1.) - o.method.ρ₂) / (T(1.) - o.method.ρ₂ ^ o.step)) * ⊙²(B))
    mul!(B, -η, /ᵉˡᵉ(C.B₁, scalar_add(racᵉˡᵉ(C.B₂), o.method.δ)))
end