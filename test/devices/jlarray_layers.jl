# Two of the seven families on a device array: a chain of dense layers, whose leaves are plain
# arrays, and a chain whose first layer is a `StiefelLayer`, so its first leaf is a manifold.
# Scalar indexing is off for the whole file, so a layer or a kernel that reads a `JLArray` one
# element at a time raises instead of answering slowly.
#
# The Stiefel row trains under a geodesic retraction. `Cayley()` is built on `lu`, JLArrays supplies no
# `lu`, and the default retraction raises `Scalar indexing is disallowed` inside the optimizer cache.

using GeometricMachineLearning
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/device.jl")

Random.seed!(123)
allowscalar(false)

@testset "$name trains two epochs on a JLArray, T = $T" for (name, arch, retraction, manifold) in (
        ("a Dense chain", Chain(Dense(4, 8, tanh), Dense(8, 4)), Cayley(), false),
        ("a StiefelLayer chain", Chain(StiefelLayer(4, 8), Dense(8, 4, tanh)), Geodesic(),
        true)),
    T in REAL_ELTYPES

    history, nn, p₀ = two_epochs(arch, T, rand(T, 4, 12, 4), 2;
        batch = Batch(6), retraction, loss = FeedForwardLoss())
    @test eltype(history) === T
    @test length(history) == 2
    @test all(isfinite, history)
    # the Stiefel leaf is a manifold of the device, and its storage is an array of `T`
    manifold && @test params(nn).L1.weight isa StiefelManifold
    @test leaves_on_device(params(nn), T)
    @test parameters_moved(p₀, params(nn))
end
