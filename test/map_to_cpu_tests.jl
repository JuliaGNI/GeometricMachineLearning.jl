# `map_to_cpu` walks a parameter set with `NeuralNetworkParameters.mapstorage`, so the structured
# types have to survive the round trip rather than come back densified. There is no GPU in CI, so
# what is pinned here is the walk and the reconstruction, not the device transfer: `Array{T}` of a
# host array is a copy, which is enough to show that every leaf was visited and rebuilt.

using GeometricMachineLearning
using GeometricMachineLearning: map_to_cpu
using LinearAlgebra
using NeuralNetworkParameters: NetworkParameters, params
using Random
using Test

import GeometricOptimizers: StiefelManifold, SymmetricMatrix, SkewSymMatrix,
                            LowerTriangular, UpperTriangular

Random.seed!(1234)

const N, n = 6, 3

@testset "a structured leaf keeps its type and its metadata" begin
    ps = NetworkParameters((
        L1 = (Y = rand(StiefelManifold{Float64}, N, n), b = randn(N)),
        L2 = (S = SymmetricMatrix(randn(n, n)), A = SkewSymMatrix(randn(n, n))),
        L3 = (L = LowerTriangular(randn(n, n)), U = UpperTriangular(randn(n, n))),
    ))

    back = map_to_cpu(ps)

    @test back isa NetworkParameters
    @test keys(back) == keys(ps)

    # every leaf comes back as the type it went in as -- this is what `rebuild` buys, and what the
    # per-type methods this replaced were for
    @test back.L1.Y isa StiefelManifold
    @test back.L2.S isa SymmetricMatrix
    @test back.L2.A isa SkewSymMatrix
    @test back.L3.L isa LowerTriangular
    @test back.L3.U isa UpperTriangular

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
    ps = NetworkParameters((L1 = (W = randn(Float32, 2, 2), S = SymmetricMatrix(randn(Float32, n, n))),))
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
