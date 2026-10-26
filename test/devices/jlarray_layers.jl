# Two of the seven families on a device array: a chain of dense layers, whose leaves are plain
# arrays, and a chain whose first layer is a `StiefelLayer`, so its first leaf is a manifold.
# Scalar indexing is off for the whole file, so a layer or a kernel that reads a `JLArray` one
# element at a time raises instead of answering slowly.
#
# The Stiefel row trains under a geodesic retraction. `Cayley` is built on `lu`, JLArrays supplies no
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

# Two epochs of `Adam` on data of the device, and the network afterwards. The loss is named because
# `train!` has no default one for a plain `Chain`.
function two_epochs(arch, ::Type{T}; retraction = Cayley()) where {T}
    Random.seed!(13)
    nn = network_on_device(arch, T)
    dl = DataLoader(JLArray(rand(T, 4, 12, 4)); suppress_info = true)
    opt = TrainingOptimizer(params(nn); algorithm = Adam(), retraction)
    history = train!(
        nn, opt, dl, Batch(6), 2; loss = FeedForwardLoss(), show_progress = false)
    history, nn
end

@testset "a Dense chain trains two epochs on a JLArray, T = $T" for T in REAL_ELTYPES
    history, nn = two_epochs(Chain(Dense(4, 8, tanh), Dense(8, 4)), T)
    @test eltype(history) === T
    @test length(history) == 2
    @test all(isfinite, history)
    @test leaves_on_device(params(nn), T)
end

@testset "a StiefelLayer chain trains two epochs on a JLArray, T = $T" for T in REAL_ELTYPES
    history, nn = two_epochs(Chain(StiefelLayer(4, 8), Dense(8, 4, tanh)), T;
        retraction = Geodesic())
    @test eltype(history) === T
    @test length(history) == 2
    @test all(isfinite, history)
    # the Stiefel leaf is a manifold of the device, and its storage is an array of `T`
    @test params(nn).L1.weight isa StiefelManifold
    @test leaves_on_device(params(nn), T)
end
