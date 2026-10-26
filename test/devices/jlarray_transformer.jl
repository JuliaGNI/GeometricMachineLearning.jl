# The standard transformer integrator on a device array. Its attention heads hold `StiefelManifold`
# weights inside a `NamedTuple` leaf, which is the one leaf shape a two-level walk does not reach, so
# the leaf assertion of the helper walks into it.
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

function two_epochs(arch, ::Type{T}) where {T}
    Random.seed!(13)
    nn = network_on_device(arch, T)
    dl = DataLoader(JLArray(rand(T, 4, 8, 12)); suppress_info = true)
    opt = TrainingOptimizer(params(nn); algorithm = Adam())
    history = train!(nn, opt, dl, Batch(6, 3), 2; show_progress = false)
    history, nn
end

@testset "StandardTransformerIntegrator trains two epochs on a JLArray, T = $T" for T in REAL_ELTYPES
    history, nn = two_epochs(StandardTransformerIntegrator(4), T)
    @test eltype(history) === T
    @test length(history) == 2
    @test all(isfinite, history)
    # one leaf of the set is a named tuple of attention heads
    @test any(x -> x isa NamedTuple, (params(nn)[n][k] for n in keys(params(nn))
    for k in keys(params(nn)[n])))
    @test leaves_on_device(params(nn), T)
end
