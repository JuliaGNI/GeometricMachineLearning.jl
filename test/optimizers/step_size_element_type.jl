# `_leaf_optim_step!` scales `direction(cache)` by `T(step_size)`, matching the three
# `_euclidean_update!` methods a few lines below it. The funnel (`_current_step_size`) always
# hands a `Float64`, no matter what `T` the parameters are, so converting first is what keeps a
# `Float32` layer scaled at `Float32` precision instead of `Float64` precision rounded back on
# write. See `CHANGELOG.md` for the fix this guards.
#
# The test is invariance rather than a fixed expected value: the leaf step must give the *same*
# result whether the funnel hands it a `Float64` step size or one already rounded to `T`. Only
# `T = Float32` can tell the two code paths apart -- `T(step_size)` is a no-op when `T` is already
# `Float64`, so a `Float64` combination is identical before and after the fix and would only pad
# the assertion count without checking anything. `Float64` is therefore not looped over here.
#
# Two calls with the *same* random draw do not reliably disagree pre-fix: whether the `Float64`-
# and `Float32`-scaled roundings land on different bit patterns depends on the actual values, so a
# fixed seed can pass pre-fix by coincidence (this happened with the seed used in an earlier
# revision of this file, for the `Adam` case only). The `seed = 2` and `step_size = 0.1` below were
# therefore checked individually, per method, against a pre-fix `origin/main` checkout (temporarily
# reverting the `T(step_size)` conversion at src/optimizers/optimizer.jl:310 and re-running): both
# `Adam` and `MomentumMethod` diverge by `2.9802322f-8` (one `Float32` ULP at this magnitude) with
# this seed, and both are exactly invariant (`diff == 0.0`) once the conversion is restored.

using GeometricMachineLearning
using GeometricOptimizers
using Test
import Random

GML = GeometricMachineLearning

function test_leaf_step_matches_prerounded_step_size(method, seed::Int, step_size::Float64)
    T = Float32
    Random.seed!(seed)
    ps_a = (weight = rand(T, 6, 6),)
    dp = (weight = rand(T, 6, 6),)
    ps_b = deepcopy(ps_a)
    λY_a = GML.GlobalSection(ps_a)
    λY_b = GML.GlobalSection(ps_b)

    cache_a = GML._make_optimizer_cache(method, ps_a)
    state_a = GML._make_optimizer_state(method, ps_a)
    cache_b = GML._make_optimizer_cache(method, ps_b)
    state_b = GML._make_optimizer_state(method, ps_b)

    # the funnel's usual `Float64` step size ...
    GML._leaf_optim_step!(
        cache_a, state_a, dp, ps_a, λY_a, method, GeometricOptimizers.cayley, step_size)
    # ... versus the same value pre-rounded to `T`.
    GML._leaf_optim_step!(
        cache_b, state_b, dp, ps_b, λY_b, method, GeometricOptimizers.cayley, T(step_size))

    @test ps_a.weight == ps_b.weight
end

test_leaf_step_matches_prerounded_step_size(GeometricOptimizers.Adam(), 2, 0.1)
test_leaf_step_matches_prerounded_step_size(GeometricOptimizers.MomentumMethod(0.5), 2, 0.1)
