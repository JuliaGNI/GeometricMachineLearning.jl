# What a training step does to a layer the loss does not read.
#
# NeuralNetworkParameters gives the gradient of such a layer as a zero leaf of the layer's own type,
# not `nothing`, and `NetworkOptimizer` hands the whole gradient to
# `GeometricOptimizers.optimization_step!`: GeometricMachineLearning skips no leaf. This script prints
# how far each leaf moves, in both precisions and under each first-order method:
#
#   - a Euclidean and a Stiefel leaf the loss never reads stay `==` their start under
#     `GradientMethod`, `MomentumMethod` and `Adam`; under `AdamWithEuclideanDecay` the Euclidean one
#     decays and the Stiefel one does not;
#   - a leaf the loss reads for the first 5 steps and not after keeps moving on its momentum under
#     `MomentumMethod`, `Adam` and `AdamWithEuclideanDecay`, and stays under `GradientMethod`;
#   - a layer without numbers, `NamedTuple()`, steps without an error.
#
# Run from the repository root, with the scripts environment:
#   julia --startup-file=no --project=scripts scripts/verification/all_zero_leaf.jl

using GeometricMachineLearning
using GeometricMachineLearning: params
# `GeometricOptimizers`' step, reached through this package: the scripts environment has no entry for it
using GeometricMachineLearning: optimization_step!
using Printf
import Random
import Zygote

const N_STEPS = 50
const N_READ = 5

maxdiff(a, b) = maximum(abs, Matrix(a) - Matrix(b))

function zero_leaf_run(T, method)
    Random.seed!(1)
    ps = NetworkParameters((read = (W = randn(T, 3, 3),),
        never_euclidean = (W = randn(T, 3, 3),),
        never_stiefel = (Y = rand(StiefelManifold{T}, 4, 2),),
        read_then_not = (W = randn(T, 3, 3),),
        empty = NamedTuple()))
    p₀ = deepcopy(ps)
    opt = NetworkOptimizer(ps; algorithm = method)
    x = randn(T, 3)
    for t in 1:N_STEPS
        loss = t ≤ N_READ ?
               (p -> sum(abs2, p.read.W * x) + sum(abs2, p.read_then_not.W * x)) :
               (p -> sum(abs2, p.read.W * x))
        dp = only(Zygote.gradient(loss, ps))
        optimization_step!(ps, opt, dp)
    end
    (never_euclidean = maxdiff(ps.never_euclidean.W, p₀.never_euclidean.W),
        never_stiefel = maxdiff(ps.never_stiefel.Y, p₀.never_stiefel.Y))
end

# The motion of the leaf that is read for `N_READ` steps, measured from step `N_READ` on.
function after_reading(T, method)
    Random.seed!(1)
    ps = NetworkParameters((
        read = (W = randn(T, 3, 3),), read_then_not = (W = randn(T, 3, 3),)))
    opt = NetworkOptimizer(ps; algorithm = method)
    x = randn(T, 3)
    both = p -> sum(abs2, p.read.W * x) + sum(abs2, p.read_then_not.W * x)
    one = p -> sum(abs2, p.read.W * x)
    for _ in 1:N_READ
        optimization_step!(ps, opt, only(Zygote.gradient(both, ps)))
    end
    W₅ = copy(ps.read_then_not.W)
    for _ in (N_READ + 1):N_STEPS
        optimization_step!(ps, opt, only(Zygote.gradient(one, ps)))
    end
    maxdiff(ps.read_then_not.W, W₅)
end

for T in (Float32, Float64)
    for method in (GradientMethod(), MomentumMethod(), Adam(), AdamWithEuclideanDecay())
        r = zero_leaf_run(T, method)
        @printf("%-8s %-26s never read: Euclidean %.2e  Stiefel %.2e | read %d steps, then not: %.2e\n",
            T, nameof(typeof(method)), r.never_euclidean,
            r.never_stiefel, N_READ,
            after_reading(T, method))
    end
end
