@doc raw"""
    BFGSOptimizer(η, δ)

Make an instance of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer. 

`η` is the *learning rate*.
`δ` is a stabilization parameter.
"""
struct BFGSOptimizer{T<:Real} <: OptimizerMethod{T}
    η::T
    δ::T

    function BFGSOptimizer(η::T = 1f-2, δ=1f-8) where T 
        new{T}(η, T(δ))
    end
end

@doc raw"""
    update!(o::Optimizer{<:BFGSOptimizer}, C, B)

Peform an update with the BFGS optimizer. 

`C` is the cache, `B` contains the gradient information (the output of [`global_rep`](@ref) in general).

First we compute the *final velocity* with
```julia
vecS = -o.method.η * C.H * vec(B)
```
and then we update `H`
```julia
C.H .= (𝕀 - ρ * SY) * C.H * (𝕀 - ρ * SY') + ρ * vecS * vecS'
```
where `SY` is `vecS * Y'` and `𝕀` is the idendity. 

# Implementation

For stability we use `δ` for computing `ρ`:
```julia
ρ = 1. / (vecS' * Y + o.method.δ)
```

This is similar to the [`AdamOptimizer`](@ref)

# Extended help 

If we have weights on a [`Manifold`](@ref) than the updates are slightly more difficult.
In this case the [`vec`](@ref) operation has to be generalized to the corresponding *global tangent space*.
"""
function update!(o::Optimizer{<:BFGSOptimizer}, C::BFGSCache, B::AbstractArray)
    T = eltype(o)
    # in the first step we compute the difference between the current and the previous mapped gradients:
    Y = vec(B - C.B)
    # compute the descent direction
    P = -C.H * vec(B)
    # compute S 
    vecS = o.method.η * P
    # store gradient
    assign!(C.B, copy(B))
    # output final velocity
    assign!(vec(B), copy(vecS))
    # compute SY and HY
    ρ = one(T) / (vecS' * Y + o.method.δ)
    SY = vecS * Y'
    𝕀 = one(SY)
    # compute H
    C.H .= (𝕀 - ρ * SY) * C.H * (𝕀 - ρ * SY') + ρ * vecS * vecS'
end