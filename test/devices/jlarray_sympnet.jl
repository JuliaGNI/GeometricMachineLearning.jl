# The three symplectic-network families on a device array: `GSympNet`, whose leaves are plain arrays
# and vectors, `LASympNet`, whose weights are `SymmetricMatrix` leaves, and
# `VolumePreservingFeedForward`, whose weights are the two strict triangular types. Scalar indexing
# is off for the whole file.
#
# None of the three holds a manifold weight, so `train!` takes its default loss for a
# `NeuralNetworkIntegrator` and a `Batch{:FeedForward}`, and the default `Cayley()` retraction is
# never reached.

using GeometricMachineLearning
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/device.jl")

Random.seed!(123)
allowscalar(false)

@testset "$name trains two epochs on a JLArray, T = $T" for (name, arch) in (
        ("GSympNet", GSympNet(4)),
        ("LASympNet", LASympNet(4)),
        ("VolumePreservingFeedForward", VolumePreservingFeedForward(4))),
    T in REAL_ELTYPES

    history, nn, p₀ = two_epochs(arch, T, rand(T, 4, 12, 4), 2; batch = Batch(6))
    @test eltype(history) === T
    @test length(history) == 2
    @test all(isfinite, history)
    @test leaves_on_device(params(nn), T)
    @test parameters_moved(p₀, params(nn))
end
