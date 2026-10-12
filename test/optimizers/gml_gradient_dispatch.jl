# What the training step sees of a gradient.
#
# `NetworkOptimizer` steps the whole parameter set with one `GeometricOptimizers` cache, so the
# gradient the cache builds its direction from is the gradient of the whole set: a parameter set of
# the network's shape, with the Riemannian gradient on a manifold leaf and the Euclidean gradient on
# a plain one.

using GeometricMachineLearning
using GeometricOptimizers: StiefelManifold, GrassmannManifold, rgrad, PrecomputedGradient
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

GML = GeometricMachineLearning

@testset "the gradient of a whole network, a manifold leaf beside a plain one ($T)" for T in REAL_ELTYPES
    for M in (StiefelManifold, GrassmannManifold)
        ps = NetworkParameters((L1 = (weight = rand(M{T}, 4, 2),),
            L2 = (W = randn(T, 3, 3), b = randn(T, 3))))
        dp = NetworkParameters((L1 = (weight = randn(T, 4, 2),),
            L2 = (W = randn(T, 3, 3), b = randn(T, 3))))
        out = PrecomputedGradient(ps, dp)(ps)

        @test out isa NetworkParameters
        @test keys(out) == keys(ps)
        @test out.L1.weight == rgrad(ps.L1.weight, dp.L1.weight)
        @test out.L2.W == dp.L2.W
        @test out.L2.b == dp.L2.b
        @test eltype(out.L1.weight) === eltype(out.L2.W) === eltype(out.L2.b) === T
    end
end

# `_get_contents` unwraps whatever shape the pullback handed back. Six methods, because a reverse pass
# produces a container or the bare `NamedTuple` it wraps, and `Zygote` may hand either back bare, in a
# one-tuple, or in a one-element vector. Each is exercised here: splitting a union into a method per
# shape means each method needs its own case, or the split trades one tested path for six untested
# ones.
@testset "_get_contents unwraps every shape a pullback returns" begin
    nt = (L1 = (W = randn(2, 2), b = randn(2)),)
    ps = NetworkParameters(nt)

    for x in (nt, ps)
        @test GML._get_contents(x) === x
        @test GML._get_contents((x,)) === x
        @test GML._get_contents([x]) === x
    end

    # more than one set is a caller error and says so, rather than silently taking the first
    @test_throws ArgumentError GML._get_contents([ps, ps])
    @test_throws ArgumentError GML._get_contents([nt, nt])
end
