# `changebackend` on the structured parameter types, which `GeometricOptimizers` 0.5 provides from
# `ext/AbstractNeuralNetworksExt.jl` -- one method on `Union{Manifold, VectorStorageMatrix,
# AbstractLieAlgHorMatrix}` that delegates to `NeuralNetworkParameters.mapstorage`.
#
# These testsets used to live in `test/hdf5_support.jl`, because the five per-type methods they
# covered used to live in this package's HDF5 extension. Both halves of that were wrong the same
# way: `changebackend` has nothing to do with HDF5, so neither the methods nor their coverage may
# depend on HDF5 being loaded. Nothing here loads it.
#
# Upstream's own `test/changebackend.jl` is the thorough test of the eight families. What is pinned
# here is that they arrive under *this* package's exported names -- the extension is keyed on
# `AbstractNeuralNetworks`, which is a hard dependency here, so it is always loaded -- and the one
# case upstream cannot reach, a whole `NeuralNetwork` built from a GML architecture.
#
# There is no second device in CI, so what a CPU -> CPU transfer pins is the walk and the
# reconstruction rather than the transfer: a leaf comes back of the same type, equal, and not the
# same array.

using GeometricMachineLearning
using LinearAlgebra: qr
using Random
using Test

import AbstractNeuralNetworks: changebackend
import GeometricOptimizers

Random.seed!(42)

const N, n = 6, 3

# All eight families, including the three that had no method here before the move: both horizontal
# lifts, and `GrassmannManifold` -- upstream dispatches on `Manifold`, so it covers that one too.
leaves = (
    stiefel   = StiefelManifold(Matrix(qr(randn(N, n)).Q)),
    grassmann = rand(GrassmannManifold{Float64}, N, n),
    symmetric = SymmetricMatrix(rand(10), 4),
    skew      = SkewSymMatrix(rand(6), 4),
    lower     = LowerTriangular(rand(6), 4),
    upper     = UpperTriangular(rand(6), 4),
    stiefhor  = StiefelLieAlgHorMatrix(SkewSymMatrix(rand(n, n)), rand(N - n, n), N, n),
    grasshor  = GrassmannLieAlgHorMatrix(rand(N - n, n), N, n),
)

@testset "every structured family keeps its type and its numbers" begin
    # one testset per family, so a failure names the leaf that failed
    for (k, x) in pairs(leaves)
        @testset "$k" begin
            y = changebackend(CPU(), x)
            @test typeof(y) == typeof(x)
            @test y ≈ x
            # a transfer copies; it does not alias the source
            @test parent(y) !== parent(x)
        end
    end
end

@testset "the metadata a structured leaf carries survives" begin
    # `n` and `N` are not in the storage, so they can only come from the prototype
    for k in (:symmetric, :skew, :lower, :upper)
        @test changebackend(CPU(), leaves[k]).n == leaves[k].n
    end
    for k in (:stiefhor, :grasshor)
        @test changebackend(CPU(), leaves[k]).N == leaves[k].N
        @test changebackend(CPU(), leaves[k]).n == leaves[k].n
    end
end

@testset "a whole NeuralNetwork moves, manifold weights and all" begin
    arch = SymplecticAutoencoder(10, 4)
    nn   = NeuralNetwork(arch)
    nn2  = changebackend(CPU(), nn)
    x    = rand(10)
    @test nn(x) ≈ nn2(x)
end

@testset "a network with a manifold weight moves without HDF5 in the picture" begin
    # The regression this move is about: while the methods sat in this package's HDF5 extension,
    # this was a `MethodError` unless HDF5 happened to be loaded. Naming the module that answers
    # says so without depending on what else the test process has loaded by now.
    Y = leaves.stiefel
    @test only(methods(changebackend, Tuple{typeof(CPU()), typeof(Y)})).module ===
          Base.get_extension(GeometricOptimizers, :AbstractNeuralNetworksExt)

    # `StiefelLayer(N, n)` is `x -> Y'x`, so it takes `N` numbers and returns `n`
    nn  = NeuralNetwork(Chain(StiefelLayer(N, n), Dense(n, n, tanh)))
    nn2 = changebackend(CPU(), nn)
    x   = rand(N)
    @test nn(x) ≈ nn2(x)
end
