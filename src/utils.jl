# Convenient structure
struct NothingFunction <: Function end
(::NothingFunction)(args...) = nothing
is_NothingFunction(f::Function) = typeof(f)==NothingFunction

struct UnknownProblem <: AbstractProblem end

const ∞ = Inf

# Functions on typple and named tuple

@inline next(i::Int,j::Int) = (i,j+1)
@inline next(i::Int) = (i+1,)

@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)


rdevelop(x) = x
rdevelop(t::Tuple{Any}) = [rdevelop(t[1])...]
rdevelop(t::Tuple) = [rdevelop(t[1])..., rdevelop(t[2:end])...]
rdevelop(t::NamedTuple) = vcat([[rdevelop(e)...] for e in t]...)

develop(x) = [x]
develop(t::Tuple{Any}) = [develop(t[1])...]
develop(t::Tuple) = [develop(t[1])..., develop(t[2:end])...]
develop(t::NamedTuple) = vcat([[develop(e)...] for e in t]...)


_tuplediff(t₁::Tuple,t₂::Tuple) = tuple(setdiff(Set(t₁),Set(t₂))...)

function apply_toNT(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(p...) for p in zip(ps...))
end

# overload norm 
_norm(dx::NT) where {AT <: AbstractArray, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}  = (norm(dx.q) + norm(dx.p)) / √2 # we need this because of a Zygote problem
_norm(dx::NamedTuple) = sum(apply_toNT(norm, dx)) / √length(dx)
_norm(A::AbstractArray) = norm(A)

# overloaded +/- operation 
_diff(dx₁::NT, dx₂::NT) where {AT <: AbstractArray, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}} = (q = dx₁.q - dx₂.q, p = dx₁.p - dx₂.p) # we need this because of a Zygote problem
_diff(dx₁::NamedTuple, dx₂::NamedTuple) = apply_toNT(_diff, dx₁, dx₂)
_diff(A::AbstractArray, B::AbstractArray) = A - B 
_add(dx₁::NamedTuple, dx₂::NamedTuple) = apply_toNT(_add, dx₁, dx₂)
_add(A::AbstractArray, B::AbstractArray) = A + B 

function add!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B) == size(C)
    C .= A + B
end

function add!(dx₁::NamedTuple, dx₂::NamedTuple, dx₃::NamedTuple)
    apply_toNT(add!, dx₁, dx₂, dx₃)
end

# Type pyracy!!
function Base.:+(a::Float64, b::Tuple{Float64})
    x, = b
    return a+x
end

# Type pyracy!!
function Base.:+(a::Vector{Float64}, b::Tuple{Float64})
    x, = b
    y, = a
    return y+x
end


# Kernel that is needed for functions relating to `SymmetricMatrix` and `SkewSymMatrix` 
@kernel function write_ones_kernel!(unit_matrix::AbstractMatrix{T}) where T
    i = @index(Global)
    unit_matrix[i, i] = one(T)
end

# overloaded similar operation to work with NamedTuples
_similar(x) = similar(x)

function _similar(x::Tuple)
    Tuple(_similar(_x) for _x in x)
end

function _similar(x::NamedTuple)
    NamedTuple{keys(x)}(_similar(values(x)))
end

# utils functions on string
function type_without_brace(var)
    type_str = string(typeof(var))
    replace(type_str, r"\{.*\}"=>"")
end

function center_align_text(text,width)
    padding = max(0, width - length(text))
    left_padding = repeat(" ",padding ÷2)
    right_padding = repeat(" ", padding - length(left_padding))
    aligned_text = left_padding * text * right_padding
    return aligned_text
end


# The following are fallback functions - maybe you want to put them into a separate file
function global_section(::AbstractVecOrMat)
    nothing
end

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
const QPT{T} = NamedTuple{(:q, :p), Tuple{AT, AT}} where {T, N, AT <: AbstractArray{T, N}}

const QPT2{T} = NamedTuple{(:q, :p), Tuple{AT₁, AT₂}} where {T, N, AT₁ <: AbstractArray{T, N}, AT₂ <: AbstractArray{T, N}}

@doc raw"""
    QPTOAT

A union of two types:
```julia
const QPTOAT = Union{QPT, AbstractArray}
```

This could be data in ``(q, p)\in\mathbb{R}^{2d}`` form or come from an arbitrary vector space.
"""
const QPTOAT{T} = Union{QPT{T}, AbstractArray{T}} where T

const QPTOAT2{T} = Union{QPT2{T}, AbstractArray{T}} where T

Base.:≈(qp₁::QPT, qp₂::QPT) = (qp₁.q ≈ qp₂.q) & (qp₁.p ≈ qp₂.p)

_eltype(x) = eltype(x)
_eltype(ps::NamedTuple) = _eltype(ps[1])
_eltype(ps::Tuple) = _eltype(ps[1])