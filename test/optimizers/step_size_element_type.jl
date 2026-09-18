# `_leaf_optim_step!` used to scale `direction(cache)` by the raw `step_size` it was handed,
# unlike its own comment: the three `_euclidean_update!` methods all write `T(step_size)`, and this
# one did not. The funnel (`_current_step_size`) always hands a `Float64`, no matter what `T` the
# parameters are, so a `Float32` layer was scaled in `Float64` precision and only rounded back to
# `Float32` on write -- a different (and more expensive) answer than scaling in `Float32` outright.
#
# The test below is invariance rather than a fixed expected value: the leaf step must give the
# *same* result whether the funnel hands it a `Float64` step size or one already rounded to `T`.
# Before the fix the two differ by a few ULPs (the `Float64`-scaled run is computed in higher
# precision); after the fix, converting to `T` first is exactly what line ~310 now does internally,
# so the two calls become identical.

using GeometricMachineLearning
using GeometricOptimizers
using Test
import Random

GML = GeometricMachineLearning
Random.seed!(7)

function test_leaf_step_matches_prerounded_step_size(method, T::DataType)
    ps_a = (weight = rand(T, 6, 6),)
    ps_b = deepcopy(ps_a)
    dp = (weight = rand(T, 6, 6),)
    λY = GML.GlobalSection(ps_a)

    cache_a = GML._make_optimizer_cache(method, ps_a)
    state_a = GML._make_optimizer_state(method, ps_a)
    cache_b = GML._make_optimizer_cache(method, ps_b)
    state_b = GML._make_optimizer_state(method, ps_b)

    # the funnel's usual `Float64` step size ...
    GML._leaf_optim_step!(
        cache_a, state_a, dp, ps_a, λY, method, GeometricOptimizers.cayley, 0.1)
    # ... versus the same value pre-rounded to `T`.
    GML._leaf_optim_step!(
        cache_b, state_b, dp, ps_b, λY, method, GeometricOptimizers.cayley, T(0.1))

    @test ps_a.weight == ps_b.weight
end

for T in (Float32, Float64)
    test_leaf_step_matches_prerounded_step_size(GeometricOptimizers.Adam(), T)
    test_leaf_step_matches_prerounded_step_size(GeometricOptimizers.MomentumMethod(0.5), T)
end
