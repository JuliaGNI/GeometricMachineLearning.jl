# Optimizer machinery on top of GeometricOptimizers.
# Kept out of `utils.jl` because it dispatches on `Manifold`, which is defined later.

# Extend GlobalSection so it works with NetworkParameters (wraps a NamedTuple).
GeometricOptimizers.GlobalSection(ps::NetworkParameters) =
    GeometricOptimizers.GlobalSection(params(ps))

# Backward-compat alias
const AbstractCache{T} = GeometricOptimizers.OptimizerCache{T}

# Gradient wrapper: stores a pre-computed Euclidean gradient and applies rgrad on manifolds.
mutable struct _GMLGradient{T, VT} <: GeometricOptimizers.Gradient{T}
    dp::VT
end

_gml_rgrad(x::Manifold, dp) = rgrad(x, dp)
_gml_rgrad(x, dp) = dp
_gml_rgrad(x::NamedTuple, dp::NamedTuple) = map(_gml_rgrad, x, dp)

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

# `AdamOptimizerWithDecay` used to be defined here, as an `OptimizerMethod` bundling Adam's `ρ₁`,
# `ρ₂`, `δ` with a learning-rate schedule `η₁`, `η₂`, `n_epochs`. GeometricOptimizers ships the same
# algorithm — the same `γ = exp(log(η₂/η₁)/n)` — split the way it belongs: the direction is an
# `Adam` method, the schedule is a `DecayingStatic` line search. Both names are imported, and
# `Optimizer` below takes a `DecayingStatic` as its `step_size`. Two packages exporting the name was
# issue B1: `using GeometricMachineLearning, GeometricOptimizers` failed outright on it.

_is_go_native_method(::GeometricOptimizers.GradientMethod) = true
_is_go_native_method(::GeometricOptimizers.MomentumMethod) = true
_is_go_native_method(::GeometricOptimizers.Adam)            = true
_is_go_native_method(::GeometricOptimizers.OptimizerMethod) = false

_adapt_method_to_T(method::GeometricOptimizers.Adam, ::Type{T}) where T =
    GeometricOptimizers.Adam(T; β₁ = T(method.β₁), β₂ = T(method.β₂), δ = T(method.δ))
_adapt_method_to_T(method::GeometricOptimizers.MomentumMethod, ::Type{T}) where T =
    GeometricOptimizers.MomentumMethod(T(method.α))
_adapt_method_to_T(method, ::Type) = method

_use_go_cache(method, x) =
    _is_go_native_method(method) && x isa GeometricOptimizers.OptimizerSolution

# A `NetworkParameters` is always a tree of layers to descend into, never a single
# `GeometricOptimizers` leaf, so its branch comes *first* — ahead of `_use_go_cache`.
#
# Today that ordering is invisible: `NetworkParameters` is not one of the types
# `GeometricOptimizers.OptimizerSolution` unions, so `_use_go_cache` is false at the root anyway and
# control reaches the container branch either way. It stops being invisible the moment
# `GeometricOptimizers` adopts the container and adds it to that union. Then `_use_go_cache` would be
# true at the *root*, and a whole network would get one cache instead of one per layer -- silently, and
# with `_leaf_optim_step!` handed the entire tree. Asking the structural question before the
# capability question is what makes the shape of the cache depend on the shape of the parameters
# rather than on which types upstream happens to accept this month.
#
# The `NamedTuple` branch stays *after* `_use_go_cache`, and that asymmetry is the point: a layer is a
# `NamedTuple` of arrays, which is exactly what one `GeometricOptimizers` cache is for. Hoisting it
# too would descend into the individual weights and give each its own `GMLEuclideanState`.
function _make_optimizer_cache(method, x)
    if x isa NetworkParameters
        NamedTuple{keys(x)}(Tuple(_make_optimizer_cache(method, x[k]) for k in keys(x)))
    elseif _use_go_cache(method, x)
        GeometricOptimizers.OptimizerCache(_adapt_method_to_T(method, parameter_eltype(x)), x)
    elseif x isa NamedTuple
        NamedTuple{keys(x)}(Tuple(_make_optimizer_cache(method, x[k]) for k in keys(x)))
    else
        GMLEuclideanState(x)
    end
end

function _make_optimizer_state(method, x)
    if x isa NetworkParameters
        NamedTuple{keys(x)}(Tuple(_make_optimizer_state(method, x[k]) for k in keys(x)))
    elseif _use_go_cache(method, x)
        GeometricOptimizers.OptimizerState(_adapt_method_to_T(method, parameter_eltype(x)), x)
    elseif x isa NamedTuple
        NamedTuple{keys(x)}(Tuple(_make_optimizer_state(method, x[k]) for k in keys(x)))
    else
        GMLEuclideanState(x)
    end
end

"""
    Optimizer(method, nn; retraction, step_size)
    Optimizer(nn; algorithm, linesearch, retraction)

Optimizer state combining a `GeometricOptimizers` method with the parameters of a neural network.

`step_size` is either a number — a fixed learning rate — or a
`GeometricOptimizers.DecayingStatic`, a learning rate that decays geometrically with the iteration
number. The second form is the one `GeometricOptimizers` uses itself, so a method paired with a
schedule splats straight in:

```julia
opt = Optimizer(nn; AdamOptimizerWithDecay(n_epochs, Float32)...)
```

# Extended help

The step size is a property of the optimizer and not of the method: the same `Adam()` trains at any
learning rate. That is the split `GeometricOptimizers` makes — the method supplies a direction, a
`SimpleSolvers.LinesearchMethod` supplies how far to go along it — and `step_size` is GML's half of
it for a training loop, which has no objective function for a real line search to evaluate.
"""
mutable struct Optimizer{MT <: GeometricOptimizers.OptimizerMethod, CT, ST, RT, SST}
    method::MT
    cache::CT
    state::ST
    retraction::RT
    step_size::SST
    iterations::Int
end

_default_step_size(::GeometricOptimizers.Adam)          = 1e-3
_default_step_size(::GeometricOptimizers.OptimizerMethod) = 1e-2

_step_size(η::Real, ::Int) = Float64(η)
# `t` and not `t - 1`: `optimization_step!` increments before it asks, so the first step of a solve
# is `α(1) = γη₁`. That is what `DecayingStatic` means by iteration `t` — `solve!` calls
# `increase_iteration_number!` before `solver_step!` — and it is what GML's own
# `AdamOptimizerWithDecay` did before the schedule moved upstream.
_step_size(ls::DecayingStatic, t::Int) = Float64(GeometricOptimizers.step_size(ls, t))

_current_step_size(opt::Optimizer, t::Int) = _step_size(opt.step_size, t)

_optimizer_step_size(η::Real) = Float64(η)
# Anything that is not a plain number goes through the same funnel as the `linesearch` keyword below,
# so that the two entry points accept the same things: `step_size = Static(α)` is a fixed learning
# rate on both, and anything else reports the `ArgumentError` that explains why a real line search
# has nothing to search along here, instead of a `MethodError` naming this helper.
_optimizer_step_size(ls) = _step_size_from_linesearch(ls)

function Optimizer(method::GeometricOptimizers.OptimizerMethod, nn::NeuralNetwork;
        retraction = GeometricOptimizers.cayley,
        step_size = _default_step_size(method))
    Optimizer(method, params(nn); retraction = retraction, step_size = step_size)
end

function Optimizer(method::GeometricOptimizers.OptimizerMethod,
        ps::Union{NamedTuple, NetworkParameters};
        retraction = GeometricOptimizers.cayley,
        step_size = _default_step_size(method))
    Optimizer(method, _make_optimizer_cache(method, ps), _make_optimizer_state(method, ps),
              retraction, _optimizer_step_size(step_size), 0)
end

# The keyword form, so that the `(algorithm, linesearch)` pairing `GeometricOptimizers` returns from
# `AdamOptimizerWithDecay` splats in unchanged. `linesearch` is the step size under the name
# upstream gives it; a `Static` carries its own `α`, which is then the fixed learning rate.
function Optimizer(nn_or_ps::Union{NeuralNetwork, NamedTuple, NetworkParameters};
        algorithm::GeometricOptimizers.OptimizerMethod,
        linesearch = nothing,
        retraction = GeometricOptimizers.cayley,
        step_size = linesearch === nothing ? _default_step_size(algorithm) :
                    _step_size_from_linesearch(linesearch))
    Optimizer(algorithm, nn_or_ps; retraction = retraction, step_size = step_size)
end

_step_size_from_linesearch(ls::DecayingStatic) = ls
_step_size_from_linesearch(ls::GeometricOptimizers.Static) = Float64(ls.α)
_step_size_from_linesearch(ls) = throw(ArgumentError(
    "`Optimizer` takes a fixed step size or a `DecayingStatic` schedule, not a $(typeof(ls)). " *
    "A training loop evaluates its loss on one batch at a time and has no objective for a line " *
    "search to search along; use `GeometricOptimizers.Optimizer` with an `OptimizerProblem` for " *
    "that."))

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
# There used to be a fourth method here, for `AdamOptimizerWithDecay`, character for character the
# `Adam` one above with `ρ₁`, `ρ₂` in place of `β₁`, `β₂`. Adam with a decaying learning rate *is*
# Adam — only `step_size` differs, and that comes in as an argument — so the `Adam` method serves it.

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
    T = parameter_eltype(ps_leaf)
    GeometricOptimizers.update!(cache, state, local_grad,
                                GeometricOptimizers.NoHessian{T}(), ps_leaf)
end

# GO-managed leaf step (manifolds, vectors, ArrayNamedTuples)
function _leaf_optim_step!(cache::GeometricOptimizers.OptimizerCache,
        state::GeometricOptimizers.OptimizerState,
        dp_leaf, ps_leaf, λY_leaf, method, retraction, step_size)
    T = parameter_eltype(ps_leaf)
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

# Recursive dispatcher over the *cache* tree.
#
# Deliberately hand-written rather than `NeuralNetworkParameters.foreachparameters`, which walks a
# parameter tree. Two reasons, and both are load-bearing:
#
#  - The recursion is keyed on `caches`, and stops where the cache stops. A cache sits at the *layer*
#    level, so a layer's `NamedTuple` of weights arrives at `_leaf_optim_step!` whole, which is what
#    one `GeometricOptimizers` cache is for. `foreachparameters` recurses on the leaf protocol and
#    would descend past the layer into the individual weights, re-pairing every cache with the wrong
#    object.
#  - `λY` is broadcast, not zipped: a single `GlobalSection` may stand in for a whole subtree, which
#    the ternary below expresses. `foreachparameters` has no such rule — it takes `values` of each
#    trailing argument, so a bare `GlobalSection` beside a `NamedTuple` of caches is a `MethodError`.
#
# The `nothing` skip is the one thing the two have in common, and it is one line here.
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

Apply one optimization step to the parameters `ps` and their gradient `dp`, with the method and step
size the [`Optimizer`](@ref) `opt` carries.

`λY` is a `GlobalSection` of `ps` (or a `NamedTuple` of them). Note that it is an *output*
here: the section the optimizer carries from step to step lives in `opt.state`, and `λY` is written
so that callers who inspect it see the updated section. It therefore has to be allocated once and
reused, not rebuilt per step -- rebuilding it costs a QR decomposition per manifold weight.

The step counter is incremented *before* the step size is read, so the first step of a run is step 1.
This matters for a decaying `step_size` and is how `GeometricOptimizers` counts too.
"""
function optimization_step!(opt::Optimizer, λY, ps, dp)
    # The increment comes *first*, so the first step of a run is step 1. It matters only for a
    # decaying `step_size`, and there it matters: reading the schedule before incrementing takes
    # `α(0) = η₁`, one whole step above what the pre-0.5 `AdamOptimizerWithDecay` took and what
    # `DecayingStatic` and `GeometricOptimizers.solve!` take. `solve!` counts the same way, by
    # calling `increase_iteration_number!` before `solver_step!`.
    opt.iterations += 1
    step = _current_step_size(opt, opt.iterations)
    _tree_optim_step!(opt.cache, opt.state, dp, ps, λY, opt.method, opt.retraction, step)
    nothing
end

check(::Optimizer) = nothing
