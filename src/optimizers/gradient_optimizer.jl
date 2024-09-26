@doc raw"""
    GradientOptimizer(η)

Make an instance of a gradient optimizer. 

This is the simplest neural network optimizer. It has no cache and computes the final velocity as:
```math
    \mathrm{velocity} \gets - \eta\nabla_\mathrm{weight}L.
```

# Implementation

The operations are done as memory efficiently as possible.
This means the provided ``\nabla_WL`` is mutated via:
```julia
rmul!(∇L, -method.η)
```
"""
struct GradientOptimizer{T<:Real} <: OptimizerMethod{T}
    η::T
    GradientOptimizer(η = 1e-2) = new{typeof(η)}(η)
end

function update!(o::Optimizer{<:GradientOptimizer}, ::GradientCache, B::AbstractVecOrMat)
    rmul!(B, -o.method.η)
end