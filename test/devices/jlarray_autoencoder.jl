# The symplectic autoencoder on a device array. Its encoder and decoder hold `StiefelManifold` and
# `PSDLayer` weights, so it trains under a geodesic retraction: `Cayley` is built on `lu`, JLArrays
# supplies none, and the default retraction raises `Scalar indexing is disallowed` inside the
# optimizer cache.
#
# The loader hands the loss a pair of tensors, because the autoencoder reconstructs its input rather
# than predicting the next state.

using GeometricMachineLearning
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/device.jl")

Random.seed!(123)
allowscalar(false)

function two_epochs(arch, ::Type{T}) where {T}
    Random.seed!(13)
    nn = network_on_device(arch, T)
    dl = DataLoader(JLArray(rand(T, 8, 12, 4)); suppress_info = true, autoencoder = true)
    opt = TrainingOptimizer(params(nn); algorithm = Adam(), retraction = Geodesic())
    history = train!(nn, opt, dl, Batch(6), 2; show_progress = false)
    history, nn
end

@testset "SymplecticAutoencoder trains two epochs on a JLArray, T = $T" for T in REAL_ELTYPES
    history, nn = two_epochs(SymplecticAutoencoder(8, 2), T)
    @test eltype(history) === T
    @test length(history) == 2
    @test all(isfinite, history)
    @test leaves_on_device(params(nn), T)
end
