# The invariant of GML #258: one pullback of a network on a device array returns every gradient leaf
# on the backend of its parameter leaf. A `ZygotePullback` that copies a gradient back to the host —
# `Array(g)` in place of `g` — breaks it silently, and the next optimizer step then reads a host
# array where the forward pass sent a device one.
#
# `test/helpers/leaves.jl` walks the two levels of a parameter set, so a gradient leaf and its
# parameter leaf are compared as a pair. The four networks below cover the four structured leaf
# families and a plain array leaf; the walk dispatches on nothing, and `parent` of a plain array is
# the array itself.
#
# Scalar indexing is off for the whole file, so a leaf that reaches the host and is read element by
# element raises instead of answering slowly.

using GeometricMachineLearning
using GeometricMachineLearning: params
using GeometricOptimizers: SymmetricMatrix, SkewSymMatrix, StrictlyLowerTriangular,
                           StrictlyUpperTriangular, StiefelManifold
using GPUArraysCore: allowscalar
using JLArrays: JLArray, JLBackend
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/device.jl")
include("../helpers/leaves.jl")

Random.seed!(123)
allowscalar(false)

# Every leaf pair of one pullback, taken the way `train!` takes it: a `ZygotePullback` of the
# network's default loss, seeded with `one(value)`. The loss sees the input twice, so it reaches
# every parameter of the network, and a leaf the pullback leaves empty would not be reached at all.
function leaf_pairs(arch, ::Type{T}, x) where {T}
    Random.seed!(17)
    nn = network_on_device(arch, T)
    input = JLArray(x)
    value, back = ZygotePullback(FeedForwardLoss())(params(nn), nn.model, (input, input))
    g = back(one(value))
    pairs = Pair{Any, Any}[]
    foreach_leaf((g_leaf, p_leaf) -> push!(pairs, g_leaf => p_leaf), g, params(nn))
    nn, pairs
end

# The two backends are one type, which is what #258 asks: the parameter leaves of these four networks
# are all arrays of the device, so a gradient leaf on `CPU` is a copy of a device leaf to the host.
function test_leaf_backends(arch, ::Type{T}, x, family) where {T}
    nn, pairs = leaf_pairs(arch, T, x)
    @test leaves_on_device(params(nn), T)
    @test !isempty(pairs)
    @test any(p -> p.second isa family, pairs)
    @test all(p -> leaf_backend(p.first) isa JLBackend, pairs)
    @test all(p -> typeof(leaf_backend(p.first)) === typeof(leaf_backend(p.second)), pairs)
    pairs
end

@testset "a StiefelManifold leaf: the gradient is a plain array of the ambient space, T = $T" for T in REAL_ELTYPES
    pairs = test_leaf_backends(
        Chain(StiefelLayer(4, 8), Dense(8, 4, tanh)), T, rand(T, 4, 6),
        StiefelManifold)
    # the `StiefelLayer`'s gradient is a plain array and not a manifold, and the `Dense` layers
    # contribute plain array leaves
    @test all(p -> p.second isa StiefelManifold ? p.first isa JLArray : true, pairs)
    @test any(p -> !(p.second isa StiefelManifold), pairs)
end

@testset "a SymmetricMatrix leaf: LASympNet, T = $T" for T in REAL_ELTYPES
    pairs = test_leaf_backends(LASympNet(4), T, rand(T, 4, 6), SymmetricMatrix)
    @test all(p -> p.second isa SymmetricMatrix ? p.first isa SymmetricMatrix : true, pairs)
end

@testset "the two triangular leaves: VolumePreservingFeedForward, T = $T" for T in REAL_ELTYPES
    pairs = test_leaf_backends(VolumePreservingFeedForward(4), T, rand(T, 4, 6),
        StrictlyLowerTriangular)
    @test any(p -> p.second isa StrictlyUpperTriangular, pairs)
    @test all(
        p -> p.second isa Union{StrictlyLowerTriangular, StrictlyUpperTriangular} ?
             typeof(p.first) === typeof(p.second) : true,
        pairs)
end

@testset "a SkewSymMatrix leaf: VolumePreservingAttention with skew_sym, T = $T" for T in REAL_ELTYPES
    pairs = test_leaf_backends(Chain(VolumePreservingAttention(4, 4; skew_sym = true)), T,
        rand(T, 4, 4, 3), SkewSymMatrix)
    @test all(p -> p.second isa SkewSymMatrix ? p.first isa SkewSymMatrix : true, pairs)
end
