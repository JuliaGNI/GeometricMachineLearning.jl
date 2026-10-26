using GeometricMachineLearning
using GeometricMachineLearning: params
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(123)

@doc raw"""
Every optimizer method has to work with the structured matrix types GML uses as *ordinary* weights:
`SymmetricMatrix` (SympNet and symplectic attention layers), `SkewSymMatrix` (volume-preserving
attention) and `StrictlyLowerTriangular`/`StrictlyUpperTriangular` (volume-preserving feedforward layers).

These have their own storage and no `setindex!`, so they need a type-preserving `similar` and
elementwise operations written on the free parameters; without them
the optimizer cache cannot even be allocated. Those live in GeometricOptimizers, which owns the
types — see `VectorStorageMatrix` there. GML used to carry its own copies of both the types and the
methods, in `src/optimizers/go_bridges.jl`.
"""
function optimizer_runs(architecture, batch, input_dim, ::Type{T}; n_epochs = 2) where {T}
    nn_and_dl() = (NeuralNetwork(architecture, T),
        DataLoader(rand(T, input_dim, 20, 5); suppress_info = true))

    for method in (Adam(), MomentumMethod(), GradientMethod())
        nn, dl = nn_and_dl()
        o = TrainingOptimizer(nn.params; algorithm = method)
        loss_array = train!(nn, o, dl, batch, n_epochs; show_progress = false)
        @test eltype(loss_array) === T
        @test length(loss_array) == n_epochs
        @test all(isfinite, loss_array)
    end

    # `AdamOptimizerWithDecay` is a `(algorithm, linesearch)` pairing rather than a method, so it
    # splats into the keywords
    nn, dl = nn_and_dl()
    o = TrainingOptimizer(nn.params; AdamOptimizerWithDecay(n_epochs)...)
    loss_array = train!(nn, o, dl, batch, n_epochs; show_progress = false)
    @test eltype(loss_array) === T
    @test length(loss_array) == n_epochs
    @test all(isfinite, loss_array)
end

# the cache has to keep the parameter's type, or the optimizer allocates dense scratch arrays and
# then fails to match them against the parameter
@testset "structured weights keep their type in `similar` ($T)" for T in REAL_ELTYPES
    for A in (rand(SymmetricMatrix{T}, 4), rand(SkewSymMatrix{T}, 4),
        rand(StrictlyLowerTriangular{T}, 4), rand(StrictlyUpperTriangular{T}, 4))
        @test eltype(A) === T
        @test typeof(similar(A)) == typeof(A)
        @test typeof(zero(A)) == typeof(A)
        @test size(similar(A)) == size(A)
    end
end

@testset "`LASympNet` (SymmetricMatrix), $T" for T in REAL_ELTYPES
    optimizer_runs(LASympNet(4), Batch(5), 4, T)
end
@testset "`VolumePreservingFeedForward` (StrictlyLower/StrictlyUpperTriangular), $T" for T in REAL_ELTYPES
    optimizer_runs(VolumePreservingFeedForward(4), Batch(5), 4, T)
end
@testset "`VolumePreservingTransformer` (SkewSymMatrix + triangular), $T" for T in REAL_ELTYPES
    optimizer_runs(VolumePreservingTransformer(4, 3), Batch(5, 3), 4, T)
end
@testset "`LinearSymplecticTransformer` (SymmetricMatrix), $T" for T in REAL_ELTYPES
    optimizer_runs(LinearSymplecticTransformer(4, 3), Batch(5, 3), 4, T)
end
