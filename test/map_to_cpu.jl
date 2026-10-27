# `map_to_cpu` walks a parameter set with `NeuralNetworkParameters.mapstorage`, so the structured
# types have to survive the round trip rather than come back densified. The GPU tests in
# `test/devices/metal.jl` do not cover `map_to_cpu`, so the device half is pinned on a `JLArray`:
# a copy of a `JLBackend` network is a network of the host, in its leaves and in its backend.
# `Array{T}` of a host array is a copy, which shows that every leaf was visited and rebuilt.

using GeometricMachineLearning
using GeometricMachineLearning: map_to_cpu
using GPUArraysCore: allowscalar
using JLArrays: JLBackend
using LinearAlgebra
using NeuralNetworkParameters: NetworkParameters, params
using Random
using Test

import GeometricOptimizers: StiefelManifold, SymmetricMatrix, SkewSymMatrix,
                            StrictlyLowerTriangular, StrictlyUpperTriangular

include("helpers/eltypes.jl")

Random.seed!(1234)

# Scalar indexing is off for the whole file, so a `_to_host` that read a device array one element at
# a time would raise rather than answer slowly.
allowscalar(false)

const N, n = 6, 3

@testset "a structured leaf keeps its type and its metadata" begin
    ps = NetworkParameters((
        L1 = (Y = rand(StiefelManifold{Float64}, N, n), b = randn(N)),
        L2 = (S = SymmetricMatrix(randn(n, n)), A = SkewSymMatrix(randn(n, n))),
        L3 = (L = StrictlyLowerTriangular(randn(n, n)),
            U = StrictlyUpperTriangular(randn(n, n)))
    ))

    back = map_to_cpu(ps)

    @test back isa NetworkParameters
    @test keys(back) == keys(ps)

    # every leaf comes back as the type it went in as -- this is what `rebuild` buys, and what the
    # per-type methods this replaced were for
    @test back.L1.Y isa StiefelManifold
    @test back.L2.S isa SymmetricMatrix
    @test back.L2.A isa SkewSymMatrix
    @test back.L3.L isa StrictlyLowerTriangular
    @test back.L3.U isa StrictlyUpperTriangular

    # the `n` a structured leaf carries is not in its storage; `rebuild` takes it from the prototype
    @test back.L2.S.n == ps.L2.S.n
    @test back.L3.U.n == ps.L3.U.n

    # and the numbers are unchanged
    @test back.L1.Y ≈ ps.L1.Y
    @test back.L1.b == ps.L1.b
    @test back.L2.S == ps.L2.S
    @test back.L3.L == ps.L3.L
end

@testset "the leaves are copies, not the same arrays" begin
    ps = NetworkParameters((L1 = (W = randn(2, 2),),))
    back = map_to_cpu(ps)
    @test back.L1.W == ps.L1.W
    @test back.L1.W !== ps.L1.W
end

@testset "element type is preserved" begin
    ps = NetworkParameters((L1 = (
        W = randn(Float32, 2, 2), S = SymmetricMatrix(randn(Float32, n, n))),))
    back = map_to_cpu(ps)
    @test eltype(back.L1.W) === Float32
    @test eltype(back.L1.S) === Float32
end

@testset "a bare NamedTuple of layers works too" begin
    ps = (L1 = (W = randn(2, 2),), L2 = (Y = rand(StiefelManifold{Float64}, N, n),))
    back = map_to_cpu(ps)
    @test back isa NamedTuple
    @test back.L2.Y isa StiefelManifold
end

@testset "a whole network keeps its architecture, model and backend" begin
    nn = NeuralNetwork(Chain(StiefelLayer(N, n), Dense(N, N, tanh)))
    back = map_to_cpu(nn)

    @test back.architecture === nn.architecture
    @test back.model === nn.model
    @test back.backend === nn.backend
    @test params(back) isa NetworkParameters
    @test keys(params(back)) == keys(params(nn))
    @test params(back).L1.weight isa StiefelManifold
end

# The copy is a network of the host, in its backend and in its fourth type parameter alike: the
# leaves are host arrays, so a backend left on the device would send the next forward pass there
# with data it cannot read.
@testset "a device network comes back on the CPU ($T)" for T in REAL_ELTYPES
    nn = NeuralNetwork(Chain(Dense(N, 3, tanh)), JLBackend(), T)
    back = map_to_cpu(nn)

    @test back.backend isa CPU
    @test typeof(back).parameters[4] === CPU
    @test params(back).L1.W isa Matrix{T}
    @test params(back).L1.b isa Vector{T}
    @test eltype(params(back).L1.W) === T
    @test eltype(params(back).L1.b) === T
    @test back.architecture === nn.architecture
end

# A host backend is a property of the caller's own network, and a copy does not change it: `CPU()`
# comes back `CPU()`, and `CPU(; static = true)` comes back static.
@testset "a host network keeps its backend and its type parameter ($T)" for T in REAL_ELTYPES
    for backend in (CPU(), CPU(; static = true))
        nn = NeuralNetwork(Chain(Dense(N, 3, tanh)), backend, T)
        back = map_to_cpu(nn)

        @test back.backend === nn.backend
        @test typeof(back).parameters[4] === CPU
        @test eltype(params(back).L1.W) === T
    end
end
