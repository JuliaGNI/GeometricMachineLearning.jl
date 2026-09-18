# `_leaf_optim_step!` scales `direction(cache)` by `T(step_size)`, matching the three
# `_euclidean_update!` methods a few lines below it. The funnel (`_current_step_size`) always
# hands a `Float64`, no matter what `T` the parameters are, so converting first is what keeps a
# `Float32` layer scaled at `Float32` precision instead of `Float64` precision rounded back on
# write. See `CHANGELOG.md` for the fix this guards.
#
# The test is invariance rather than a fixed expected value: the leaf step must give the *same*
# result whether the funnel hands it a `Float64` step size or one already rounded to `T`. Only
# `T = Float32` can tell the two calls apart -- `T(step_size)` is a no-op when `T` is already
# `Float64`, so a `Float64` combination holds whatever the conversion does and checks nothing.
# `Float64` is therefore not looped over here.
#
# The `seed = 2` and the `step_size = 0.1` are chosen rather than arbitrary. Whether a `Float64`-
# and a `Float32`-scaled direction round to different bit patterns depends on the actual values,
# so for most seeds the two calls agree however the step size is scaled, and the test then passes
# without discriminating. With these values, and with the conversion dropped, both `Adam` and
# `MomentumMethod` separate by `2.9802322f-8` -- one `Float32` ULP at this magnitude -- and each
# method is checked on its own.

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
