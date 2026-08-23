# Convenient structure
struct NothingFunction <: Function end
(::NothingFunction)(args...) = nothing
is_NothingFunction(f::Function) = typeof(f) == NothingFunction

struct UnknownProblem <: AbstractProblem end

const ∞ = Inf

# Functions on typple and named tuple

@inline next(i::Int, j::Int) = (i, j + 1)
@inline next(i::Int) = (i + 1,)

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

_tuplediff(t₁::Tuple, t₂::Tuple) = tuple(setdiff(Set(t₁), Set(t₂))...)

function apply_toNT(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(p...) for p in zip(ps...))
end

# overload norm
function _norm(dx::NT) where {
        AT <: AbstractArray, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    (norm(dx.q) + norm(dx.p)) / √2
end # we need this because of a Zygote problem
_norm(dx::NamedTuple) = sum(apply_toNT(norm, dx)) / √length(dx)
_norm(A::AbstractArray) = norm(A)

# overloaded +/- operation
function _diff(dx₁::NT,
        dx₂::NT) where {AT <: AbstractArray, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    (q = dx₁.q - dx₂.q, p = dx₁.p - dx₂.p)
end # we need this because of a Zygote problem
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
    return a + x
end

# Type pyracy!!
function Base.:+(a::Vector{Float64}, b::Tuple{Float64})
    x, = b
    y, = a
    return y + x
end

# Kernel that is needed for functions relating to `SymmetricMatrix` and `SkewSymMatrix`
@kernel function write_ones_kernel!(unit_matrix::AbstractMatrix{T}) where {T}
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
    replace(type_str, r"\{.*\}" => "")
end

function center_align_text(text, width)
    padding = max(0, width - length(text))
    left_padding = repeat(" ", padding ÷ 2)
    right_padding = repeat(" ", padding - length(left_padding))
    aligned_text = left_padding * text * right_padding
    return aligned_text
end

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
const QPT{T} = NamedTuple{(:q, :p), Tuple{AT, AT}} where {T, N, AT <: AbstractArray{T, N}}

@doc raw"""
    QPT2

[`QPT`](@ref) with the number of dimensions of the two arrays fixed, but their types allowed to
differ. A `Chain` that splits an input array into `q` and `p` produces views of different types, so
the layers of a parameter-dependent network dispatch on this rather than on `QPT`.
"""
const QPT2{T, N} = NamedTuple{(:q, :p), Tuple{AT₁, AT₂}} where {T, N, AT₁ <: AbstractArray{T, N}, AT₂ <: AbstractArray{T, N}}

@doc raw"""
    QPTOAT

A union of two types:
```julia
const QPTOAT = Union{QPT, AbstractArray}
```

This could be data in ``(q, p)\in\mathbb{R}^{2d}`` form or come from an arbitrary vector space.
"""
const QPTOAT{T} = Union{QPT{T}, AbstractArray{T}} where {T}

@doc raw"""
    QPTOAT2

[`QPTOAT`](@ref) with the number of dimensions of the arrays fixed:

```julia
const QPTOAT2 = Union{QPT2, AbstractArray}
```
"""
const QPTOAT2{T, N} = Union{QPT2{T, N}, AbstractArray{T, N}} where {T, N}

Base.:≈(qp₁::QPT, qp₂::QPT) = (qp₁.q ≈ qp₂.q) & (qp₁.p ≈ qp₂.p)

@doc raw"""
    _flatten_system_parameters(parameters)
    _flatten_system_parameters(T, parameters)

Flatten the parameters of the *system* (not of the network) into a vector, together with the
`NeuralNetworkParameters.ParameterLayout` that puts such a vector back into the original shape.

The parameter-dependent architectures — [`GeneralizedHamiltonianArchitecture`](@ref) and the layers
it is built from — feed the system parameters to the network as extra input components, so they have
to be a vector. `NullParameters` flattens to an empty one, which makes the parameter-free case fall
out of the same code path.

The layout is a *value*, not a closure, so a layer can store it in a field and stay inferable.
"""
_flatten_system_parameters(parameters::NamedTuple) = flatten(parameters)
_flatten_system_parameters(::NullParameters) = flatten(NamedTuple())
_flatten_system_parameters(::Type{T}, parameters::NamedTuple) where {T} = flatten(T, parameters)
_flatten_system_parameters(::Type{T}, ::NullParameters) where {T} = flatten(T, NamedTuple())

"""
    _unwrap_gradient(dp)

Strip the `NetworkParameters` wrappers and the `(params = …,)` layers out of a gradient, so
that it has the same shape as the parameters it belongs to.

`Zygote` differentiates *through* the `NetworkParameters` struct, so the gradient of a
parameter set comes back as a `NamedTuple` with a single `params` field. [`_get_params`](@ref) undoes
that at the top level. The parameter-dependent architectures nest — a `SymplecticEuler` layer
holds the parameters of a whole sub-network — so the unwrapping has to recurse.
"""
_unwrap_gradient(dp) = dp
_unwrap_gradient(dp::NetworkParameters) = _unwrap_gradient(params(dp))
_unwrap_gradient(dp::NamedTuple{(:params,)}) = _unwrap_gradient(dp.params)
_unwrap_gradient(dp::NamedTuple) = map(_unwrap_gradient, dp)

_eltype(x) = eltype(x)
_eltype(ps::NamedTuple) = _eltype(ps[1])
_eltype(ps::Tuple) = _eltype(ps[1])
_eltype(ps::NetworkParameters) = _eltype(params(ps)[1])

# `ParametricDataLoader` stores one `NamedTuple` of system parameters per trajectory, and they all
# have to agree with the element type of the data.
function _eltype(parameters::AbstractVector{<:NamedTuple})
    T = _eltype(first(parameters))
    for p in parameters
        _eltype(p) == T || error("The parameters do not all have the same element type.")
    end
    T
end

# `size` that also works on `(q, p)` data, where the first axis is the concatenation of the two.
_size(x) = size(x)
function _size(qp::QPT)
    q_size = _size(qp.q)
    p_size = _size(qp.p)
    @assert q_size == p_size
    (2q_size[1], q_size[2:end]...)
end

_size(x, a::Integer) = size(x, a)
function _size(qp::QPT, a::Integer)
    q_size = _size(qp.q, a)
    p_size = _size(qp.p, a)
    @assert q_size == p_size
    a == 1 ? 2q_size : q_size
end
