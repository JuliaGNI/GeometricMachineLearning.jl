# Optimizer machinery on top of GeometricOptimizers.
# Kept out of `utils.jl` because it dispatches on `Manifold`, which is defined later.

# Gradient wrapper: stores a pre-computed Euclidean gradient and applies rgrad on manifolds.
mutable struct _GMLGradient{T, VT} <: GeometricOptimizers.Gradient{T}
    dp::VT
end

_gml_rgrad(x::Manifold, dp) = rgrad(x, dp)
_gml_rgrad(x, dp) = dp
# `mapparameters` and not `map`: it recurses on the branches, so `_gml_rgrad` is only ever called on
# leaves, and it rebuilds in the shape of its *first* argument -- a container, which is what
# `GeometricOptimizers._copyto!(gradient_array(cache), ·)` has a method for. It also normalises its
# trailing arguments, so `dp` may stay the plain `NamedTuple` the pullback produced.
_gml_rgrad(x::NetworkParameters, dp) = mapparameters(_gml_rgrad, x, dp)

(g::_GMLGradient{T})(x::NetworkParameters{T}) where {T} = _gml_rgrad(x, g.dp)
(g::_GMLGradient{T})(x::AbstractArray{T}) where {T} = g.dp

# `Manifold` and not `AbstractArray` alone, because the two are otherwise **ambiguous** and not merely
# overlapping. `_GMLGradient{T}` is a `GeometricOptimizers.Gradient{T}`, `Manifold{T}` is an
# `AbstractMatrix{T}`, and `GeometricOptimizers` has `(::Gradient{T})(::Manifold{T})`; so on a bare
# manifold neither `(::_GMLGradient{T})(::AbstractArray{T})` above nor that one is more specific, and
# the call is a run-time `MethodError: ... is ambiguous`. This method is more specific than both.
#
# The answer it gives is this package's: the gradient is already computed, so the projection is
# `rgrad(x, dp)` and not upstream's `rgrad(x, reshape(grad(vec(x)), ...))`, which would evaluate an
# inner gradient this functor does not have. `_gml_rgrad` is where that lives.
#
# `_make_optimizer_cache` below asks `_is_layer` first and so keeps a bare `Manifold` from arriving
# here at all. That ordering is a correctness fix in its own right -- one cache per layer rather than
# per weight -- but it must not be the *only* thing standing between a caller and an ambiguity.
(g::_GMLGradient{T})(x::Manifold{T}) where {T} = _gml_rgrad(x, g.dp)

# A bare `AbstractVector`, for the same reason as the `Manifold` method above: `SimpleSolvers` has
# `(::Gradient{T})(::AbstractVector{T})`, `_GMLGradient{T} <: GeometricOptimizers.Gradient{T}`, and
# without a method of its own `(::_GMLGradient{T})(::AbstractArray{T})` above and that one are
# equally specific on a `Vector` -- the call is a run-time `MethodError: ... is ambiguous`. This
# method is more specific than both: `_GMLGradient{T} <: Gradient{T}` on the first argument and
# `AbstractVector{T} <: AbstractArray{T}` on the second. The answer is unchanged from the
# `AbstractArray{T}` method above -- the gradient is already Euclidean, so it is `g.dp`.
(g::_GMLGradient{T})(x::AbstractVector{T}) where {T} = g.dp

# State for Euclidean (non-manifold) parameters.
mutable struct GMLEuclideanState{T, AT <: AbstractArray{T}}
    iterations::Int
    m₁::AT
    m₂::AT
end
function GMLEuclideanState(x::AbstractArray{T}) where {T}
    GMLEuclideanState{T, typeof(x)}(0, zero(x), zero(x))
end

# Adam with a decaying learning rate is GeometricOptimizers', and split the way it belongs: the
# direction is an `Adam` method, the schedule is a `DecayingStatic` line search. Both names are
# imported, and `Optimizer` below takes a `DecayingStatic` as its `step_size`.

_is_go_native_method(::GeometricOptimizers.GradientMethod) = true
_is_go_native_method(::GeometricOptimizers.MomentumMethod) = true
_is_go_native_method(::GeometricOptimizers.Adam) = true
# `ScalarMomentAdam` is driven through the same cache as the other three: `_go_update_leaf!` hands
# `update!` the method (it is a `FirstOrderMethodWithState`) and `GeometricOptimizers.sync_state!`
# carries its scalar second moment back into the state. Without this arm it fell through to
# `GMLEuclideanState`, which is coordinate-wise Adam on the ambient array -- not the method, and not
# on the manifold.
_is_go_native_method(::GeometricOptimizers.ScalarMomentAdam) = true
_is_go_native_method(::GeometricOptimizers.OptimizerMethod) = false
# A `CompositeMethod` is whatever it selects, and what it selects depends on the leaf -- so this
# question cannot be answered about the composite itself, only about a leaf's method. Every call
# site below resolves the leaf's method with `leafmethod` before asking, and this arm is what says
# so if one is ever added that does not.
_is_go_native_method(::GeometricOptimizers.CompositeMethod) = throw(ArgumentError(
    "`_is_go_native_method` is a question about a leaf's method; resolve the composite with " *
    "`GeometricOptimizers.leafmethod(method, leaf)` first"))

function _adapt_method_to_T(method::GeometricOptimizers.Adam, ::Type{T}) where {T}
    GeometricOptimizers.Adam(T; β₁ = T(method.β₁), β₂ = T(method.β₂), δ = T(method.δ))
end
function _adapt_method_to_T(method::GeometricOptimizers.MomentumMethod, ::Type{T}) where {T}
    GeometricOptimizers.MomentumMethod(T(method.α))
end
# As for `Adam`: `ScalarMomentAdam` carries parameters of its own and is not converted by
# `Optimizer`, so it has to be constructed with the element type of the parameters or its cache
# constructor raises. `ambient_norm` is a mode and not a number, so it is carried across unchanged.
function _adapt_method_to_T(method::GeometricOptimizers.ScalarMomentAdam, ::Type{T}) where {T}
    GeometricOptimizers.ScalarMomentAdam(T; β₁ = T(method.β₁), β₂ = T(method.β₂),
        δ = T(method.δ), ambient_norm = method.ambient_norm)
end
_adapt_method_to_T(method, ::Type) = method

# Whether one `GeometricOptimizers` cache covers `x` whole.
#
# A **layer** is a flat `NamedTuple` of arrays, and it is exactly what one cache is for. A `NamedTuple`
# whose values are branches is a *subtree* and gets one cache per layer instead, which is why the test
# is flatness and not merely "a `NamedTuple`".
#
# The rule lives here rather than upstream because a whole set of parameters reaches
# `GeometricOptimizers` only as a `NetworkParameters`, and a container says nothing about how deep the
# tree beneath it is. Splitting a network into layers is this package's decision, so this package makes
# it; [`_as_go_solution`](@ref) then does the wrap.
#
# A layer whose weights do not share an element type is still one cache, with `T` the promotion over
# them. `_adapt_method_to_T` reads that promotion, so the cache is built for the element type the
# layer actually has rather than for one its weights are required to agree on.
# `!isempty` and not `all` alone: `all` over an empty collection is vacuously true, so without it an
# empty set would be one cache's worth -- and `OptimizerCache` would then be asked for the element
# type of a set that has no leaves to promote. An empty set is zero caches' worth, so it descends and
# yields an empty tree of caches, which is what the step below iterates over zero times.
_is_layer(x::NamedTuple) = _all_leaves(values(x))
_is_layer(x::NetworkParameters) = _all_leaves(values(x))
_is_layer(_) = false

_all_leaves(vs) = !isempty(vs) && all(v -> v isa AbstractArray, vs)

function _use_go_cache(method, x)
    _is_go_native_method(method) &&
        (x isa GeometricOptimizers.OptimizerSolution || _is_layer(x))
end

"""
    _as_go_solution(x)
    _as_go_solution(x, method)

`x` in the shape `GeometricOptimizers` takes a solution in for `method`.

A layer given as a bare `NamedTuple` is wrapped; anything already in the right shape — a container, a
bare `Manifold`, an `AbstractArray` — is passed through. The wrap **shares the leaf arrays**, so an
in-place optimizer step writes through to the network's own weights and nothing has to be copied back.

The two-argument form additionally *unwraps* for a method whose scope is a single leaf rather than a
set of them. `GeometricOptimizers.ScalarMomentAdam` is the one that is: its second moment is a
scalar, which is a statement about one manifold and means nothing pooled across a set, so it rejects
a container even when that container holds exactly one Stiefel weight. Which methods those are is
upstream's `accepts_parameter_set` and not a list of type names kept here; *when* to unwrap — a
layer of exactly one weight, and an error naming the layer otherwise — is this package's rule,
because the grouping into layers is.
"""
_as_go_solution(x::NetworkParameters) = x
_as_go_solution(x) = _is_layer(x) ? NetworkParameters(x) : x

function _as_go_solution(x, method)
    GeometricOptimizers.accepts_parameter_set(method) && return _as_go_solution(x)
    _single_weight(x, method)
end

"""
    _as_go_leaf(x, method)

The *unwrap* half of [`_as_go_solution`](@ref), for the gradient and the section tree that travel
beside a solution.

Neither wants the wrap — `_gml_rgrad` normalises a plain `NamedTuple` gradient itself, and a section
tree is copied into as it stands — but both have to follow the solution through the unwrap, so that
a method whose scope is a single weight is handed that weight's gradient and that weight's section
rather than the layer's.
"""
function _as_go_leaf(x, method)
    GeometricOptimizers.accepts_parameter_set(method) ? x : _single_weight(x, method)
end

# The unwrap, and the error that stands in for it. A method of single-leaf scope selected for a layer
# that holds more than one weight is a mistake in the *selector*, and saying which layer and which
# method is what makes it findable; without this it surfaces several frames down as upstream's
# scope `ArgumentError`, which knows nothing about layers.
_single_weight(x::Union{NetworkParameters, NamedTuple}, method) = _single_weight(values(x), x, method)
_single_weight(x, ::Any) = x

function _single_weight(weights, layer, method)
    length(weights) == 1 || throw(ArgumentError(
        "$(nameof(typeof(method))) takes a single weight as its solution, but the layer it was " *
        "selected for holds $(length(weights)) ($(join(keys(layer), ", "))); select a method " *
        "that accepts a parameter set for this layer"))
    only(weights)
end

# A `NetworkParameters` is always a tree of layers to descend into, never a single
# `GeometricOptimizers` leaf, so its branch comes *first* — ahead of `_use_go_cache`.
#
# That ordering is **load-bearing**. `NetworkParameters` *is* one of the types
# `GeometricOptimizers.OptimizerSolution` unions, so `_use_go_cache` is true at the root: without the
# test above, a whole network would get one cache instead of one per layer -- silently, and with
# `_leaf_optim_step!` handed the entire tree. Asking the structural question before the capability
# question is what makes the shape of the cache depend on the shape of the parameters rather than on
# which types upstream happens to accept.
#
# One cache for the whole network is the better end state, and it is **not** what this does.
# `_GMLGradient` does take a `NetworkParameters`, but that is one wrapped *layer* and not the tree,
# because the container branch above splits the network first. Getting to one cache means dropping
# `_tree_optim_step!` and handing the whole tree over, which is a change of behaviour -- one
# `GlobalSection` tree and one `Q` across every layer -- so it wants its own release. Nothing here
# depends on it.
#
# **"Is this one cache's worth?" is asked before "is this a tree?"**, and a *flat* set answers yes to
# both. `_is_layer` is that question: a set whose values are all leaves is a layer, whether it arrives
# wrapped or as a branch, and one `GeometricOptimizers` cache is exactly what a layer is for. Asking
# the tree question first would descend into such a set and give every individual weight its own cache
# -- which for a manifold weight is not merely wasteful but wrong, since a bare `Manifold` then reaches
# the gradient functor where a whole layer should have.
#
# The tree branch comes second and covers both carriers, because a network is a tree of layers whether
# it is wrapped or not.
# Whether `x` is something to descend into rather than something one `GeometricOptimizers` object is
# built for. A flat set is *not* a branch: it is a layer, and one cache is what a layer is for.
_is_branch(x) = (x isa NetworkParameters || x isa NamedTuple) && !_is_layer(x)

# The method that steps `x`, which for a `GeometricOptimizers.CompositeMethod` depends on `x` and for
# every other method is the method itself. A branch is passed through untouched: it is descended
# into, and its layers are what get asked, so a selector is never shown a whole network.
#
# **This is why the layer test now comes first in the three branches below.** A composite cannot
# answer `_is_go_native_method` about itself -- that is a property of the method it selects, which
# depends on the leaf -- so asking it before establishing that `x` is a layer would raise on every
# tree root. The ordering the original comment is about, structure before capability, is unchanged:
# `_is_layer` is still the first question, and a flat set is still one cache.
_leaf_method(method, x) = _is_branch(x) ? method : GeometricOptimizers.leafmethod(method, x)

function _make_optimizer_cache(method, x)
    leaf_method = _leaf_method(method, x)
    if _is_layer(x) && _is_go_native_method(leaf_method)
        GeometricOptimizers.OptimizerCache(
            _adapt_method_to_T(leaf_method, parameter_eltype(x)),
            _as_go_solution(x, leaf_method))
    elseif x isa NetworkParameters || x isa NamedTuple
        NamedTuple{keys(x)}(Tuple(_make_optimizer_cache(method, x[k]) for k in keys(x)))
    elseif _use_go_cache(leaf_method, x)
        GeometricOptimizers.OptimizerCache(
            _adapt_method_to_T(leaf_method, parameter_eltype(x)), x)
    else
        GMLEuclideanState(x)
    end
end

function _make_optimizer_state(method, x)
    leaf_method = _leaf_method(method, x)
    if _is_layer(x) && _is_go_native_method(leaf_method)
        GeometricOptimizers.OptimizerState(
            _adapt_method_to_T(leaf_method, parameter_eltype(x)),
            _as_go_solution(x, leaf_method))
    elseif x isa NetworkParameters || x isa NamedTuple
        NamedTuple{keys(x)}(Tuple(_make_optimizer_state(method, x[k]) for k in keys(x)))
    elseif _use_go_cache(leaf_method, x)
        GeometricOptimizers.OptimizerState(
            _adapt_method_to_T(leaf_method, parameter_eltype(x)), x)
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

_default_step_size(::GeometricOptimizers.Adam) = 1e-3
_default_step_size(::GeometricOptimizers.OptimizerMethod) = 1e-2

_step_size(η::Real, ::Int) = Float64(η)
# `t` and not `t - 1`: `optimization_step!` increments before it asks, so the first step of a solve
# is `α(1) = γη₁`. That is what `DecayingStatic` means by iteration `t`: `solve!` calls
# `increase_iteration_number!` before `solver_step!`.
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
        ps::NetworkParameters;
        retraction = GeometricOptimizers.cayley,
        step_size = _default_step_size(method))
    Optimizer(method, _make_optimizer_cache(method, ps), _make_optimizer_state(method, ps),
        retraction, _optimizer_step_size(step_size), 0)
end

# The keyword form, so that the `(algorithm, linesearch)` pairing `GeometricOptimizers` returns from
# `AdamOptimizerWithDecay` splats in unchanged. `linesearch` is the step size under the name
# upstream gives it; a `Static` carries its own `α`, which is then the fixed learning rate.
function Optimizer(nn_or_ps::Union{NeuralNetwork, NetworkParameters};
        algorithm::GeometricOptimizers.OptimizerMethod,
        linesearch = nothing,
        retraction = GeometricOptimizers.cayley,
        step_size = linesearch === nothing ? _default_step_size(algorithm) :
                    _step_size_from_linesearch(linesearch))
    Optimizer(algorithm, nn_or_ps; retraction = retraction, step_size = step_size)
end

_step_size_from_linesearch(ls::DecayingStatic) = ls
_step_size_from_linesearch(ls::GeometricOptimizers.Static) = Float64(ls.α)
function _step_size_from_linesearch(ls)
    throw(ArgumentError(
        "`Optimizer` takes a fixed step size or a `DecayingStatic` schedule, not a $(typeof(ls)). " *
        "A training loop evaluates its loss on one batch at a time and has no objective for a line " *
        "search to search along; use `GeometricOptimizers.Optimizer` with an `OptimizerProblem` for " *
        "that."))
end

# Euclidean update rules
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState, ::GeometricOptimizers.GradientMethod, step_size) where {T}
    x .-= T(step_size) .* dx
end
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState{T}, method::GeometricOptimizers.MomentumMethod, step_size) where {T}
    # `p ← αp + ∇L`, the classic momentum recursion. The decay belongs on `p` and not on `∇L`:
    # `p ← p + α∇L` is an undamped accumulator that grows without bound for a constant gradient
    # instead of saturating at `∇L/(1 - α)`. Same recursion as GO's `update!(::MomentumState, ...)`.
    state.m₁ .= T(method.α) .* state.m₁ .+ dx
    x .-= T(step_size) .* state.m₁
end
function _euclidean_update!(x::AbstractArray{T}, dx::AbstractArray,
        state::GMLEuclideanState{T}, method::GeometricOptimizers.Adam, step_size) where {T}
    t = state.iterations
    _t = t + 1
    β₁, β₂, δ = T(method.β₁), T(method.β₂), T(method.δ)
    # the first factor is `(β - β^t)/(1 - β^t)`, not `β/(1 - β^t)`: the latter is ~100x too large
    # at t = 2 and compounds every step, which inflates the second moment until the update vanishes
    fac₁₁ = (β₁-β₁^_t)/(1-β₁^_t)
    fac₁₂ = (1-β₁)/(1-β₁^_t)
    fac₂₁ = (β₂-β₂^_t)/(1-β₂^_t)
    fac₂₂ = (1-β₂)/(1-β₂^_t)
    state.m₁ .= fac₁₁ .* state.m₁ .+ fac₁₂ .* dx
    state.m₂ .= fac₂₁ .* state.m₂ .+ fac₂₂ .* dx .^ 2
    x .-= T(step_size) .* state.m₁ ./ (sqrt.(state.m₂) .+ δ)
end
# `_euclidean_update!` has three methods and not four: Adam with a decaying learning rate *is*
# Adam — only `step_size` differs, and that comes in as an argument — so the `Adam` method above
# serves it too.

# The split upstream's `solver_step!` makes, and made here for the same reason: a
# `FirstOrderMethodWithState` builds its direction out of state of its own and so needs the *method*,
# and everything else needs the Hessian and has none. Written as that union rather than as one arm
# per method, so that a method joining it upstream -- `ScalarMomentAdam` did -- does not leave a
# stale list behind here silently taking the Hessian branch instead.
function _go_update_leaf!(cache, state, local_grad,
        method::GeometricOptimizers.FirstOrderMethodWithState, ps_leaf)
    GeometricOptimizers.update!(cache, state, local_grad, method, ps_leaf)
end

function _go_update_leaf!(cache, state, local_grad,
        method::GeometricOptimizers.OptimizerMethod, ps_leaf)
    T = parameter_eltype(ps_leaf)
    GeometricOptimizers.update!(cache, state, local_grad,
        GeometricOptimizers.NoHessian{T}(), ps_leaf)
end

# GO-managed leaf step (manifolds, vectors, whole layers)
#
# `ps` and not `ps_leaf` from here on: a layer is wrapped for the same reason its cache is, and the
# wrap shares the arrays, so `_copyto!(ps, solution(cache))` below writes into the weights the caller
# holds. A bare `Manifold` or `AbstractArray` passes through untouched.
function _leaf_optim_step!(cache::GeometricOptimizers.OptimizerCache,
        state::GeometricOptimizers.OptimizerState,
        dp_leaf, ps_leaf, λY_leaf, method, retraction, step_size)
    T = parameter_eltype(ps_leaf)
    adapted = _adapt_method_to_T(GeometricOptimizers.leafmethod(method, ps_leaf), T)
    ps = _as_go_solution(ps_leaf, adapted)
    # The gradient and the section tree take the *unwrap* half only: `_gml_rgrad` normalises a plain
    # `NamedTuple` gradient itself and the section tree is copied into as it stands, so neither wants
    # the wrap. Both have to follow the solution through the unwrap, though -- a method whose scope
    # is a single weight is handed that weight's gradient and that weight's section.
    dp = _as_go_leaf(dp_leaf, adapted)
    λY = _as_go_leaf(λY_leaf, adapted)
    local_grad = _GMLGradient{T, typeof(dp)}(dp)
    state.iterations += 1
    _go_update_leaf!(cache, state, local_grad, adapted, ps)
    GeometricOptimizers._rmul!(GeometricOptimizers.direction(cache), T(step_size))
    GeometricOptimizers.update_section!(GeometricOptimizers.section(cache),
        GeometricOptimizers.section(state),
        GeometricOptimizers.direction(cache),
        retraction)
    GeometricOptimizers._copyto!(GeometricOptimizers.solution(cache),
        GeometricOptimizers.section(cache))
    GeometricOptimizers._copyto!(ps, GeometricOptimizers.solution(cache))
    GeometricOptimizers._copyto!(λY, GeometricOptimizers.section(cache))
    # `section(cache)` is `update_section!(section(state), direction, retraction)`, so copying it is
    # the same thing as retracting a second time -- and a retraction on a manifold is `O(N³)` where
    # the copy is `O(N²)`.
    GeometricOptimizers._copyto!(GeometricOptimizers.section(state),
        GeometricOptimizers.section(cache))
    # What a method carries out of a step -- a momentum, a pair of moments -- and whether the cache's
    # copy is the one to keep is the method's own knowledge, and upstream's since 0.8. This was a
    # chain of `isa` tests over the state types here, which is a list of methods maintained in the
    # package that does not define them: `ScalarMomentAdam` was simply missing from it.
    GeometricOptimizers.sync_state!(state, cache, adapted)
    nothing
end

# Euclidean leaf step (plain AbstractArray params)
function _leaf_optim_step!(cache::GMLEuclideanState, state::GMLEuclideanState,
        dp_leaf, ps_leaf, λY_leaf, method, retraction, step_size)
    # The composite is resolved here too: this arm is reached for a leaf whose selected method has
    # no `GeometricOptimizers` cache, and `_euclidean_update!` dispatches on that method.
    _euclidean_update!(ps_leaf, dp_leaf, state,
        GeometricOptimizers.leafmethod(method, ps_leaf), step_size)
    state.iterations += 1
    nothing
end

# Whether `λY` is a tree of sections to descend into, or a single `GlobalSection` standing in for a
# whole subtree. Named rather than written as an `isa` union: it is one question, asked once, and the
# answer decides whether a layer gets its own section or the whole tree. A section tree is a plain
# `NamedTuple` today, because `GeometricOptimizers`' `GlobalSection(::NetworkParameters)` unwraps the
# container; the container arm is here so that a change there is a `MethodError` and not a silent
# hand-off of the whole tree to every layer.
_is_section_tree(::NamedTuple) = true
_is_section_tree(::NetworkParameters) = true
_is_section_tree(_) = false

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
#
# The `λY` test names both container types for the same reason `_make_optimizer_cache` asks the
# structural question first: a section tree is something to descend into, whichever type carries it.
# Only a `NamedTuple` ever arrives: `GeometricOptimizers`' own
# `GlobalSection(ps::NetworkParameters) = GlobalSection(params(ps))` unwraps the container, so a
# section tree is a plain `NamedTuple` whether the parameters are wrapped or not. Should that ever
# return a *container* of sections instead, `isa NamedTuple` alone would be false here and every layer
# would be handed the whole tree instead of its own section.
function _tree_optim_step!(caches, states, dp, ps, λY, method, retraction, step_size)
    if caches isa NamedTuple
        for k in keys(caches)
            dp_k = dp[k]
            dp_k === nothing && continue
            λY_k = _is_section_tree(λY) ? λY[k] : λY
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
    # `α(0) = η₁`, one whole step above what `DecayingStatic` and `GeometricOptimizers.solve!`
    # take. `solve!` counts the same way, by calling `increase_iteration_number!` before
    # `solver_step!`.
    opt.iterations += 1
    step = _current_step_size(opt, opt.iterations)
    _tree_optim_step!(opt.cache, opt.state, dp, ps, λY, opt.method, opt.retraction, step)
    nothing
end

check(::Optimizer) = nothing
