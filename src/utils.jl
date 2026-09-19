# overload norm
function _norm(dx::NT) where {
        AT <: AbstractArray, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    T = eltype(dx.q)
    (norm(dx.q) + norm(dx.p)) / √(T(2))
end # we need this because of a Zygote problem
function _norm(dx::NamedTuple)
    n = sum(map(norm, dx))
    n / √(typeof(n)(length(dx)))
end
_norm(A::AbstractArray) = norm(A)

# overloaded - operation
function _diff(dx₁::NT,
        dx₂::NT) where {AT <: AbstractArray, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    (q = dx₁.q - dx₂.q, p = dx₁.p - dx₂.p)
end # we need this because of a Zygote problem
_diff(dx₁::NamedTuple, dx₂::NamedTuple) = map(_diff, dx₁, dx₂)
_diff(A::AbstractArray, B::AbstractArray) = A - B

# `global_section(::AbstractVecOrMat) = nothing` used to be defined here, identically to
# GeometricOptimizers' own fallback. It is imported now.

"""
    QPT

The type for data in ``(q, p)`` coordinates. It encompasses various array types.

# Examples

```jldoctest
using GeometricMachineLearning: QPT

# allocate two vectors
data1 = (q = rand(5), p = rand(5))

# allocate two matrices
data2 = (q = rand(5, 4), p = rand(5, 4))

# allocate two tensors
data3 = (q = rand(5, 4, 2), p = rand(5, 4, 2))

(typeof(data1) <: QPT, typeof(data2) <: QPT, typeof(data3) <: QPT)

# output

(true, true, true)
```

We can also do:

```jldoctest
using GeometricMachineLearning: QPT, PoissonTensor

𝕁 = PoissonTensor(4)
qp = (q = [1, 2], p = [3, 4])

𝕁 * qp

# output

(q = [3, 4], p = [-1, -2])
```

"""
const QPT{T} = NamedTuple{(:q, :p), Tuple{AT, AT}} where {T, AT <: AbstractArray{T}}

@doc raw"""
    QPTOAT

A union of two types:
```julia
const QPTOAT = Union{QPT, AbstractArray}
```

This could be data in ``(q, p)\in\mathbb{R}^{2d}`` form or come from an arbitrary vector space.
"""
const QPTOAT{T} = Union{QPT{T}, AbstractArray{T}} where {T}
