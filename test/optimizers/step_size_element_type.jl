# The step scales in the parameters' own element type. `NetworkOptimizer` hands `linesearch` to
# `GeometricOptimizers.TrainingOptimizer`, which converts it once to the element type of the
# parameters, so a `Float64` step size on a `Float32` network gives the same step as the same value
# rounded to `Float32` first.
#
# Only `T = Float32` can tell the two apart -- for `Float64` the rounding is a no-op -- so `Float64`
# is not looped over here.

using GeometricMachineLearning
using GeometricOptimizers: optimization_step!
using Test
import Random

function test_step_matches_prerounded_step_size(method, seed::Int, step_size::Float64)
    T = Float32
    Random.seed!(seed)
    ps_a = NetworkParameters((L1 = (weight = rand(T, 6, 6),),))
    dp = NetworkParameters((L1 = (weight = rand(T, 6, 6),),))
    ps_b = deepcopy(ps_a)

    opt_a = NetworkOptimizer(ps_a; algorithm = method, linesearch = step_size)
    opt_b = NetworkOptimizer(ps_b; algorithm = method, linesearch = T(step_size))
    @test typeof(opt_a.optimizer.linesearch) == typeof(opt_b.optimizer.linesearch)

    optimization_step!(ps_a, opt_a, dp)
    optimization_step!(ps_b, opt_b, dp)

    @test ps_a.L1.weight isa Matrix{T}
    @test ps_a.L1.weight == ps_b.L1.weight
end

test_step_matches_prerounded_step_size(Adam(), 2, 0.1)
test_step_matches_prerounded_step_size(MomentumMethod(; α = 0.5), 2, 0.1)
