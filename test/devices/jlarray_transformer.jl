# The standard transformer integrator on a device array. Its attention heads are a `NamedTuple` of
# plain arrays, which is the one leaf shape a two-level walk does not reach, so the leaf assertion of
# the helper walks into it.
#
# The batch is the transformer's: `Batch(n, L)` draws minibatches of `n` sequences of `L` time steps,
# and the second axis of a `DataLoader`'s input is the time axis, eight steps here, so `Batch(6, 3)`
# takes windows of three of them.

using GeometricMachineLearning
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/device.jl")

Random.seed!(123)
allowscalar(false)

@testset "StandardTransformerIntegrator trains two epochs on a JLArray, T = $T" for T in REAL_ELTYPES
    history, nn, p₀ = two_epochs(StandardTransformerIntegrator(4), T, rand(T, 4, 8, 12), 2;
        batch = Batch(6, 3))
    @test eltype(history) === T
    @test length(history) == 2
    @test all(isfinite, history)
    # one leaf of the set is a named tuple of attention heads
    @test any(x -> x isa NamedTuple, (params(nn)[n][k] for n in keys(params(nn))
    for k in keys(params(nn)[n])))
    @test leaves_on_device(params(nn), T)
    @test parameters_moved(p₀, params(nn))
end
