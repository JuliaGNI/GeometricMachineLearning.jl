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
_eltype(ps::NeuralNetworkParameters) = _eltype(params(ps)[1])

# Extend GlobalSection so it works with NeuralNetworkParameters (wraps a NamedTuple).
GeometricOptimizers.GlobalSection(ps::NeuralNetworkParameters) =
    GeometricOptimizers.GlobalSection(params(ps))

# Backward-compat alias
const AbstractCache{T} = GeometricOptimizers.OptimizerCache{T}

# Gradient wrapper: stores a pre-computed Euclidean gradient and applies rgrad on manifolds.
mutable struct _GMLGradient{T, VT} <: GeometricOptimizers.Gradient{T}
    dp::VT
end
(g::_GMLGradient{T})(x::GeometricOptimizers.ArrayNamedTuple{T}) where {T} =
    GeometricOptimizers.apply_toNT(rgrad, x, g.dp)
(g::_GMLGradient{T})(x::AbstractArray{T}) where {T} = g.dp

# State for Euclidean (non-manifold) parameters.
mutable struct GMLEuclideanState{T, AT<:AbstractArray{T}}
    iterations::Int
    m₁::AT
    m₂::AT
end
GMLEuclideanState(x::AbstractArray{T}) where T =
    GMLEuclideanState{T, typeof(x)}(0, zero(x), zero(x))

# Adam with exponential learning-rate decay.
struct AdamOptimizerWithDecay{T<:Real} <: GeometricOptimizers.OptimizerMethod
    η₁::T; η₂::T; ρ₁::T; ρ₂::T; δ::T; γ::T; n_epochs::Int
    function AdamOptimizerWithDecay(n_epochs::Int, η₁=1f-2, η₂=1f-6,
            ρ₁=9f-1, ρ₂=9.9f-1, δ=1f-8; T=typeof(η₁))
        γ = exp(log(η₂/η₁) / n_epochs)
        new{T}(T(η₁), T(η₂), T(ρ₁), T(ρ₂), T(δ), T(γ), n_epochs)
    end
end

_is_go_native_method(::GeometricOptimizers.GradientMethod) = true
_is_go_native_method(::GeometricOptimizers.MomentumMethod) = true
_is_go_native_method(::GeometricOptimizers.Adam)            = true
_is_go_native_method(::GeometricOptimizers.OptimizerMethod) = false

_adapt_method_to_T(method::GeometricOptimizers.Adam, ::Type{T}) where T =
    GeometricOptimizers.Adam(T(method.η), T(method.β₁), T(method.β₂), T(method.δ))
_adapt_method_to_T(method::GeometricOptimizers.MomentumMethod, ::Type{T}) where T =
    GeometricOptimizers.MomentumMethod(T(method.α))
_adapt_method_to_T(method, ::Type) = method

_use_go_cache(method, x) =
    _is_go_native_method(method) && x isa GeometricOptimizers.OptimizerSolution

function _make_optimizer_cache(method, x)
    if _use_go_cache(method, x)
        GeometricOptimizers.OptimizerCache(_adapt_method_to_T(method, _eltype(x)), x)
    elseif x isa NamedTuple || x isa NeuralNetworkParameters
        NamedTuple{keys(x)}(Tuple(_make_optimizer_cache(method, x[k]) for k in keys(x)))
    else
        GMLEuclideanState(x)
    end
end

function _make_optimizer_state(method, x)
    if _use_go_cache(method, x)
        GeometricOptimizers.OptimizerState(method, x)
    elseif x isa NamedTuple || x isa NeuralNetworkParameters
        NamedTuple{keys(x)}(Tuple(_make_optimizer_state(method, x[k]) for k in keys(x)))
    else
        GMLEuclideanState(x)
    end
end

mutable struct Optimizer{MT <: GeometricOptimizers.OptimizerMethod, CT, ST, RT}
    method::MT
    cache::CT
    state::ST
    retraction::RT
    step_size::Float64
    iterations::Int
end

_default_step_size(method::GeometricOptimizers.Adam)  = Float64(method.η)
_default_step_size(method::AdamOptimizerWithDecay)     = Float64(method.η₁)
_default_step_size(::GeometricOptimizers.OptimizerMethod) = 1e-2

_current_step_size(opt::Optimizer, ::Int) = opt.step_size
_current_step_size(opt::Optimizer{<:AdamOptimizerWithDecay}, t::Int) =
    Float64(opt.method.η₁ * opt.method.γ^t)

function Optimizer(method::GeometricOptimizers.OptimizerMethod, nn::NeuralNetwork;
        retraction = GeometricOptimizers.cayley,
        step_size::Real = _default_step_size(method))
    ps = params(nn)
    Optimizer(method, _make_optimizer_cache(method, ps), _make_optimizer_state(method, ps),
              retraction, Float64(step_size), 0)
end

function Optimizer(method::GeometricOptimizers.OptimizerMethod,
        ps::Union{NamedTuple, NeuralNetworkParameters};
        retraction = GeometricOptimizers.cayley,
        step_size::Real = _default_step_size(method))
    Optimizer(method, _make_optimizer_cache(method, ps), _make_optimizer_state(method, ps),
              retraction, Float64(step_size), 0)
end

# Euclidean update rules
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState, ::GeometricOptimizers.GradientMethod, step_size) where T
    x .-= T(step_size) .* dx
end
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState{T}, method::GeometricOptimizers.MomentumMethod, step_size) where T
    x .-= T(step_size) .* (dx .+ state.m₁)
    state.m₁ .+= T(method.α) .* dx
end
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState{T}, method::GeometricOptimizers.Adam, step_size) where T
    t = state.iterations; _t = t + 1
    β₁, β₂, δ = T(method.β₁), T(method.β₂), T(method.δ)
    fac₁₁ = β₁/(1-β₁^_t); fac₁₂ = (1-β₁)/(1-β₁^_t)
    fac₂₁ = β₂/(1-β₂^_t); fac₂₂ = (1-β₂)/(1-β₂^_t)
    state.m₁ .= fac₁₁ .* state.m₁ .+ fac₁₂ .* dx
    state.m₂ .= fac₂₁ .* state.m₂ .+ fac₂₂ .* dx .^ 2
    x .-= T(step_size) .* state.m₁ ./ (sqrt.(state.m₂) .+ δ)
end
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState{T}, method::AdamOptimizerWithDecay, step_size) where T
    t = state.iterations; _t = t + 1
    ρ₁, ρ₂, δ = T(method.ρ₁), T(method.ρ₂), T(method.δ)
    fac₁₁ = ρ₁/(1-ρ₁^_t); fac₁₂ = (1-ρ₁)/(1-ρ₁^_t)
    fac₂₁ = ρ₂/(1-ρ₂^_t); fac₂₂ = (1-ρ₂)/(1-ρ₂^_t)
    state.m₁ .= fac₁₁ .* state.m₁ .+ fac₁₂ .* dx
    state.m₂ .= fac₂₁ .* state.m₂ .+ fac₂₂ .* dx .^ 2
    x .-= T(step_size) .* state.m₁ ./ (sqrt.(state.m₂) .+ δ)
end

# GO-managed leaf step (manifolds, vectors, ArrayNamedTuples)
function _leaf_optim_step!(cache::GeometricOptimizers.OptimizerCache,
        state::GeometricOptimizers.OptimizerState,
        dp_leaf, ps_leaf, λY_leaf, method, retraction, step_size)
    T = _eltype(ps_leaf)
    local_grad = _GMLGradient{T, typeof(dp_leaf)}(dp_leaf)
    adapted = _adapt_method_to_T(method, T)
    if adapted isa GeometricOptimizers.Adam
        GeometricOptimizers.update!(cache, state, local_grad, adapted, ps_leaf)
    else
        GeometricOptimizers.update!(cache, state, local_grad,
                                     GeometricOptimizers.NoHessian{T}(), ps_leaf)
    end
    GeometricOptimizers._rmul!(GeometricOptimizers.direction(cache), step_size)
    GeometricOptimizers.update_section!(GeometricOptimizers.section(cache),
                                         GeometricOptimizers.section(state),
                                         GeometricOptimizers.direction(cache),
                                         retraction)
    GeometricOptimizers._copyto!(GeometricOptimizers.solution(cache),
                                  GeometricOptimizers.section(cache))
    GeometricOptimizers._copyto!(ps_leaf, GeometricOptimizers.solution(cache))
    GeometricOptimizers._copyto!(λY_leaf, GeometricOptimizers.section(cache))
    GeometricOptimizers.update_section!(GeometricOptimizers.section(state),
                                         GeometricOptimizers.section(state),
                                         GeometricOptimizers.direction(cache),
                                         retraction)
    if state isa GeometricOptimizers.AdamState
        GeometricOptimizers._copyto!(GeometricOptimizers.first_moment(state),
                                      GeometricOptimizers.first_moment(cache))
        GeometricOptimizers._copyto!(GeometricOptimizers.second_moment(state),
                                      GeometricOptimizers.second_moment(cache))
    elseif state isa GeometricOptimizers.MomentumState
        GeometricOptimizers._add!(GeometricOptimizers.momentum(state),
                                   GeometricOptimizers._mul(adapted.α,
                                       GeometricOptimizers.gradient_array(cache)))
    end
    state.iterations += 1
    nothing
end

# Euclidean leaf step (plain AbstractArray params)
function _leaf_optim_step!(cache::GMLEuclideanState, state::GMLEuclideanState,
        dp_leaf, ps_leaf, λY_leaf, method, retraction, step_size)
    _euclidean_update!(ps_leaf, dp_leaf, state, method, step_size)
    state.iterations += 1
    nothing
end

# Recursive dispatcher over the parameter tree
function _tree_optim_step!(caches, states, dp, ps, λY, method, retraction, step_size)
    if caches isa NamedTuple
        for k in keys(caches)
            dp_k = dp[k]
            dp_k === nothing && continue
            λY_k = λY isa NamedTuple ? λY[k] : λY
            _tree_optim_step!(caches[k], states[k], dp_k, ps[k], λY_k,
                              method, retraction, step_size)
        end
    else
        _leaf_optim_step!(caches, states, dp, ps, λY, method, retraction, step_size)
    end
    nothing
end

function optimization_step!(opt::Optimizer, λY, ps, dp)
    step = _current_step_size(opt, opt.iterations)
    _tree_optim_step!(opt.cache, opt.state, dp, ps, λY, opt.method, opt.retraction, step)
    opt.iterations += 1
    nothing
end

check(::Optimizer) = nothing
