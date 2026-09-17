using GeometricMachineLearning
using GeometricMachineLearning: params
using Test
import Random

Random.seed!(123)

@doc raw"""
Every optimizer method has to work with the structured matrix types GML uses as *ordinary* weights:
`SymmetricMatrix` (SympNet and symplectic attention layers), `SkewSymMatrix` (volume-preserving
attention) and `LowerTriangular`/`UpperTriangular` (volume-preserving feedforward layers).

These have their own storage and no `setindex!`, so they need a type-preserving `similar` and
elementwise `_add!`/`_rac!`/`_square!`/`_div!`/`_rmul!` written on the free parameters; without them
the optimizer cache cannot even be allocated. Those live in GeometricOptimizers, which owns the
types — see `VectorStorageMatrix` there. GML used to carry its own copies of both the types and the
methods, in `src/optimizers/go_bridges.jl`.
"""
function optimizer_runs(architecture, batch, input_dim; T = Float64, n_epochs = 2)
    nn_and_dl() = (NeuralNetwork(architecture, T),
        DataLoader(rand(T, input_dim, 20, 5); suppress_info = true))

    for method in (AdamOptimizer(), MomentumOptimizer(), GradientOptimizer())
        nn, dl = nn_and_dl()
        o = Optimizer(method, nn)
        loss_array = o(nn, dl, batch, n_epochs; show_progress = false)
        @test length(loss_array) == n_epochs
        @test all(isfinite, loss_array)
    end

    # `AdamOptimizerWithDecay` is a `(algorithm, linesearch)` pairing rather than a method, so it
    # goes in through the keyword constructor
    nn, dl = nn_and_dl()
    o = Optimizer(nn; AdamOptimizerWithDecay(n_epochs, T)...)
    loss_array = o(nn, dl, batch, n_epochs; show_progress = false)
    @test length(loss_array) == n_epochs
    @test all(isfinite, loss_array)
end

# the cache has to keep the parameter's type, or the optimizer allocates dense scratch arrays and
# then fails to match them against the parameter
@testset "structured weights keep their type in `similar`" begin
    for A in (rand(SymmetricMatrix{Float64}, 4), rand(SkewSymMatrix{Float64}, 4),
        rand(LowerTriangular{Float64}, 4), rand(UpperTriangular{Float64}, 4))
        @test typeof(similar(A)) == typeof(A)
        @test typeof(zero(A)) == typeof(A)
        @test size(similar(A)) == size(A)
    end
end

@testset "`LASympNet` (SymmetricMatrix)" begin
    optimizer_runs(LASympNet(4), Batch(5), 4)
end
@testset "`VolumePreservingFeedForward` (Lower/UpperTriangular)" begin
    optimizer_runs(VolumePreservingFeedForward(4), Batch(5), 4)
end
@testset "`VolumePreservingTransformer` (SkewSymMatrix + triangular)" begin
    optimizer_runs(VolumePreservingTransformer(4, 3), Batch(5, 3), 4)
end
@testset "`LinearSymplecticTransformer` (SymmetricMatrix)" begin
    optimizer_runs(LinearSymplecticTransformer(4, 3), Batch(5, 3), 4)
end
