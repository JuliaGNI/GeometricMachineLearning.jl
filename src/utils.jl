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

Base.:≈(qp₁::QPT, qp₂::QPT) = (qp₁.q ≈ qp₂.q) & (qp₁.p ≈ qp₂.p)

_eltype(x) = eltype(x)
_eltype(ps::NamedTuple) = _eltype(ps[1])
_eltype(ps::Tuple) = _eltype(ps[1])

# Backward-compat alias: the old GML name for the abstract cache type.
const AbstractCache{T} = GeometricOptimizers.OptimizerCache{T}

# A mutable gradient wrapper that returns a precomputed gradient.
# Subtypes SimpleSolvers.Gradient (accessible as GeometricOptimizers.Gradient) so it works
# with GeometricOptimizers' cache update! methods.
mutable struct _GMLGradient{T, VT} <: GeometricOptimizers.Gradient{T}
    dp::VT  # pre-computed Euclidean gradient (NamedTuple or AbstractArray)
end
# When called with parameters, apply rgrad and return the Riemannian gradient.
(g::_GMLGradient{T})(x::GeometricOptimizers.ArrayNamedTuple{T}) where {T} =
    GeometricOptimizers.apply_toNT(rgrad, x, g.dp)
(g::_GMLGradient{T})(x::AbstractArray{T}) where {T} = g.dp  # euclidean: no rgrad needed

"""
    Optimizer

GML's neural-network optimizer. Wraps a GeometricOptimizers method together with its
corresponding cache, state, and retraction.
"""
mutable struct Optimizer{MT <: GeometricOptimizers.OptimizerMethod, CT, ST, RT}
    method::MT
    cache::CT
    state::ST
    retraction::RT
    _grad::_GMLGradient  # mutable gradient reference used by optimization_step!
end

function Optimizer(method::GeometricOptimizers.OptimizerMethod, nn::NeuralNetwork;
        retraction = GeometricOptimizers.cayley)
    ps = params(nn)
    T  = eltype(ps[1])
    cache = GeometricOptimizers.OptimizerCache(method, ps)
    state = GeometricOptimizers.OptimizerState(method, ps)
    grad  = _GMLGradient{T, typeof(ps)}(ps)  # dummy initial dp (gets overwritten)
    Optimizer(method, cache, state, retraction, grad)
end

# Convenience constructor that accepts a raw params NamedTuple directly.
function Optimizer(method::GeometricOptimizers.OptimizerMethod, ps::Union{NamedTuple, NeuralNetworkParameters};
        retraction = GeometricOptimizers.cayley)
    T  = eltype(ps[1])
    cache = GeometricOptimizers.OptimizerCache(method, ps)
    state = GeometricOptimizers.OptimizerState(method, ps)
    grad  = _GMLGradient{T, typeof(ps)}(ps)  # dummy initial dp (gets overwritten)
    Optimizer(method, cache, state, retraction, grad)
end

"""
    optimization_step!(opt, λY, ps, dp)

Perform one optimizer step given a pre-computed Euclidean gradient `dp`.

- `λY`  — the `GlobalSection` of the current parameters `ps`.
- `ps`  — the current parameter NamedTuple (modified in-place).
- `dp`  — the Euclidean gradient returned by Zygote.
"""
function optimization_step!(opt::Optimizer, λY, ps, dp)
    opt._grad.dp = dp  # inject the pre-computed gradient

    # Step 1: update the cache (computes gradient in Lie algebra, then direction)
    if opt.method isa GeometricOptimizers.Adam
        GeometricOptimizers.update!(opt.cache, opt.state, opt._grad, opt.method, ps)
    else
        hess = GeometricOptimizers.NoHessian{eltype(ps[1])}()
        GeometricOptimizers.update!(opt.cache, opt.state, opt._grad, hess, ps)
    end

    # Step 2: apply retraction — update section(cache) from section(state) + direction
    GeometricOptimizers.update_section!(
        GeometricOptimizers.section(opt.cache),
        GeometricOptimizers.section(opt.state),
        GeometricOptimizers.direction(opt.cache),
        opt.retraction
    )

    # Step 3: copy new manifold point from cache section → cache solution → ps and λY
    GeometricOptimizers._copyto!(GeometricOptimizers.solution(opt.cache),
                                  GeometricOptimizers.section(opt.cache))
    GeometricOptimizers._copyto!(ps, GeometricOptimizers.solution(opt.cache))
    GeometricOptimizers._copyto!(λY, GeometricOptimizers.section(opt.cache))

    # Step 4: advance the state's section to the new position (needed for next step)
    GeometricOptimizers.update_section!(
        GeometricOptimizers.section(opt.state),
        GeometricOptimizers.section(opt.state),
        GeometricOptimizers.direction(opt.cache),
        opt.retraction
    )
    opt.state.iterations += 1
    nothing
end

# Convenience: allow check(opt) to print nothing (backward compat)
check(::Optimizer) = nothing
