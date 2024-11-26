@doc raw"""
    MomentumOptimizer(η, α)

Make an instance of the momentum optimizer.

The momentum optimizer is similar to the [`GradientOptimizer`](@ref).
It however has a nontrivial cache that stores past history (see [`MomentumCache`](@ref)).
The cache is updated via:
```math
    B^{\mathrm{cache}} \gets \alpha{}B^{\mathrm{cache}} + \nabla_\mathrm{weights}L
```
and then the final velocity is computed as
```math
    \mathrm{velocity} \gets  - \eta{}B^{\mathrm{cache}}.
```

# Implementation

To save memory the *velocity* is stored in the input ``\nabla_WL``.
This is similar to the case of the [`GradientOptimizer`](@ref).
"""
struct MomentumOptimizer{T<:Real} <: OptimizerMethod{T}
    η::T
    α::T
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α)
end

#update for weights
function update!(o::Optimizer{<:MomentumOptimizer}, C::MomentumCache, B::AbstractVecOrMat)
    add!(C.B, o.method.α*C.B, B)
    mul!(B, -o.method.η, C.B)
end