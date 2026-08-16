# Optimizer machinery on top of GeometricOptimizers.
# Kept out of `utils.jl` because it dispatches on `Manifold`, which is defined later.

# Extend GlobalSection so it works with NeuralNetworkParameters (wraps a NamedTuple).
GeometricOptimizers.GlobalSection(ps::NeuralNetworkParameters) =
    GeometricOptimizers.GlobalSection(params(ps))

# Backward-compat alias
const AbstractCache{T} = GeometricOptimizers.OptimizerCache{T}

# Gradient wrapper: stores a pre-computed Euclidean gradient and applies rgrad on manifolds.
mutable struct _GMLGradient{T, VT} <: GeometricOptimizers.Gradient{T}
    dp::VT
end

_gml_rgrad(x::Manifold, dp) = rgrad(x, dp)
_gml_rgrad(x, dp) = dp
_gml_rgrad(x::NamedTuple, dp::NamedTuple) =
    GeometricOptimizers.apply_toNT(_gml_rgrad, x, dp)

(g::_GMLGradient{T})(x::GeometricOptimizers.ArrayNamedTuple{T}) where {T} =
    _gml_rgrad(x, g.dp)
(g::_GMLGradient{T})(x::AbstractArray{T}) where {T} = g.dp

# State for Euclidean (non-manifold) parameters.
mutable struct GMLEuclideanState{T, AT<:AbstractArray{T}}
    iterations::Int
    m₁::AT
    m₂::AT
end
GMLEuclideanState(x::AbstractArray{T}) where T =
    GMLEuclideanState{T, typeof(x)}(0, zero(x), zero(x))

"""Adam optimizer method with exponential learning-rate decay."""
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
# `AdamOptimizerWithDecay` differs from `Adam` only in the step size, which GML supplies separately
# through `_current_step_size`, so it uses GO's Adam cache and state like any other Adam.
_is_go_native_method(::AdamOptimizerWithDecay)              = true
_is_go_native_method(::GeometricOptimizers.OptimizerMethod) = false

_adapt_method_to_T(method::GeometricOptimizers.Adam, ::Type{T}) where T =
    GeometricOptimizers.Adam(T; β₁ = T(method.β₁), β₂ = T(method.β₂), δ = T(method.δ))
_adapt_method_to_T(method::GeometricOptimizers.MomentumMethod, ::Type{T}) where T =
    GeometricOptimizers.MomentumMethod(T(method.α))
_adapt_method_to_T(method::AdamOptimizerWithDecay, ::Type{T}) where T =
    GeometricOptimizers.Adam(T; β₁ = T(method.ρ₁), β₂ = T(method.ρ₂), δ = T(method.δ))
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
        GeometricOptimizers.OptimizerState(_adapt_method_to_T(method, _eltype(x)), x)
    elseif x isa NamedTuple || x isa NeuralNetworkParameters
        NamedTuple{keys(x)}(Tuple(_make_optimizer_state(method, x[k]) for k in keys(x)))
    else
        GMLEuclideanState(x)
    end
end

"""Optimizer state combining a GeometricOptimizers method with GML parameters."""
mutable struct Optimizer{MT <: GeometricOptimizers.OptimizerMethod, CT, ST, RT}
    method::MT
    cache::CT
    state::ST
    retraction::RT
    step_size::Float64
    iterations::Int
end

_default_step_size(::GeometricOptimizers.Adam)          = 1e-3
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
    # `p ← αp + ∇L`, the classic momentum recursion. The decay belongs on `p` and not on `∇L`:
    # `p ← p + α∇L` is an undamped accumulator that grows without bound for a constant gradient
    # instead of saturating at `∇L/(1 - α)`. Same recursion as GO's `update!(::MomentumState, ...)`.
    state.m₁ .= T(method.α) .* state.m₁ .+ dx
    x .-= T(step_size) .* state.m₁
end
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState{T}, method::GeometricOptimizers.Adam, step_size) where T
    t = state.iterations; _t = t + 1
    β₁, β₂, δ = T(method.β₁), T(method.β₂), T(method.δ)
    # the first factor is `(β - β^t)/(1 - β^t)`, not `β/(1 - β^t)`: the latter is ~100x too large
    # at t = 2 and compounds every step, which inflates the second moment until the update vanishes
    fac₁₁ = (β₁-β₁^_t)/(1-β₁^_t); fac₁₂ = (1-β₁)/(1-β₁^_t)
    fac₂₁ = (β₂-β₂^_t)/(1-β₂^_t); fac₂₂ = (1-β₂)/(1-β₂^_t)
    state.m₁ .= fac₁₁ .* state.m₁ .+ fac₁₂ .* dx
    state.m₂ .= fac₂₁ .* state.m₂ .+ fac₂₂ .* dx .^ 2
    x .-= T(step_size) .* state.m₁ ./ (sqrt.(state.m₂) .+ δ)
end
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState{T}, method::AdamOptimizerWithDecay, step_size) where T
    t = state.iterations; _t = t + 1
    ρ₁, ρ₂, δ = T(method.ρ₁), T(method.ρ₂), T(method.δ)
    # see the note in the `Adam` method above
    fac₁₁ = (ρ₁-ρ₁^_t)/(1-ρ₁^_t); fac₁₂ = (1-ρ₁)/(1-ρ₁^_t)
    fac₂₁ = (ρ₂-ρ₂^_t)/(1-ρ₂^_t); fac₂₂ = (1-ρ₂)/(1-ρ₂^_t)
    state.m₁ .= fac₁₁ .* state.m₁ .+ fac₁₂ .* dx
    state.m₂ .= fac₂₁ .* state.m₂ .+ fac₂₂ .* dx .^ 2
    x .-= T(step_size) .* state.m₁ ./ (sqrt.(state.m₂) .+ δ)
end

function _go_update_leaf!(cache, state, local_grad,
        method::GeometricOptimizers.Adam, ps_leaf)
    GeometricOptimizers.update!(cache, state, local_grad, method, ps_leaf)
end

function _go_update_leaf!(cache, state, local_grad,
        method::GeometricOptimizers.MomentumMethod, ps_leaf)
    GeometricOptimizers.update!(cache, state, local_grad, method, ps_leaf)
end

function _go_update_leaf!(cache, state, local_grad,
        method::GeometricOptimizers.OptimizerMethod, ps_leaf)
    T = _eltype(ps_leaf)
    GeometricOptimizers.update!(cache, state, local_grad,
                                GeometricOptimizers.NoHessian{T}(), ps_leaf)
end

# GO-managed leaf step (manifolds, vectors, ArrayNamedTuples)
function _leaf_optim_step!(cache::GeometricOptimizers.OptimizerCache,
        state::GeometricOptimizers.OptimizerState,
        dp_leaf, ps_leaf, λY_leaf, method, retraction, step_size)
    T = _eltype(ps_leaf)
    local_grad = _GMLGradient{T, typeof(dp_leaf)}(dp_leaf)
    adapted = _adapt_method_to_T(method, T)
    state.iterations += 1
    _go_update_leaf!(cache, state, local_grad, adapted, ps_leaf)
    GeometricOptimizers._rmul!(GeometricOptimizers.direction(cache), step_size)
    GeometricOptimizers.update_section!(GeometricOptimizers.section(cache),
                                         GeometricOptimizers.section(state),
                                         GeometricOptimizers.direction(cache),
                                         retraction)
    GeometricOptimizers._copyto!(GeometricOptimizers.solution(cache),
                                  GeometricOptimizers.section(cache))
    GeometricOptimizers._copyto!(ps_leaf, GeometricOptimizers.solution(cache))
    GeometricOptimizers._copyto!(λY_leaf, GeometricOptimizers.section(cache))
    # `section(cache)` is `update_section!(section(state), direction, retraction)`, so copying it is
    # the same thing as retracting a second time -- and a retraction on a manifold is `O(N³)` where
    # the copy is `O(N²)`.
    GeometricOptimizers._copyto!(GeometricOptimizers.section(state),
                                  GeometricOptimizers.section(cache))
    if state isa GeometricOptimizers.AdamState
        GeometricOptimizers._copyto!(GeometricOptimizers.first_moment(state),
                                      GeometricOptimizers.first_moment(cache))
        GeometricOptimizers._copyto!(GeometricOptimizers.second_moment(state),
                                      GeometricOptimizers.second_moment(cache))
    elseif state isa GeometricOptimizers.MomentumState
        # `p ← αp + ∇L`; see the note in `_euclidean_update!` for the momentum method. This has to
        # match what `update!(::MomentumCache, ...)` anticipated when it formed the direction.
        GeometricOptimizers._rmul!(GeometricOptimizers.momentum(state), adapted.α)
        GeometricOptimizers._add!(GeometricOptimizers.momentum(state),
                                   GeometricOptimizers.gradient_array(cache))
    end
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

"""
    optimization_step!(opt, λY, ps, dp)

Apply one optimization step to the parameters `ps` and their gradient `dp`.

`λY` is a `GlobalSection` of `ps` (or a `NamedTuple` of them). Note that it is an *output*
here: the section the optimizer carries from step to step lives in `opt.state`, and `λY` is written
so that callers who inspect it see the updated section. It therefore has to be allocated once and
reused, not rebuilt per step -- rebuilding it costs a QR decomposition per manifold weight.
"""
function optimization_step!(opt::Optimizer, λY, ps, dp)
    step = _current_step_size(opt, opt.iterations)
    _tree_optim_step!(opt.cache, opt.state, dp, ps, λY, opt.method, opt.retraction, step)
    opt.iterations += 1
    nothing
end

check(::Optimizer) = nothing
