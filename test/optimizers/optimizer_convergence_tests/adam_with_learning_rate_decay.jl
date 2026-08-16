
using GeometricMachineLearning
using GeometricMachineLearning: ResNetLayer, params
using LinearAlgebra: I
using Test
import Random

Random.seed!(123)

const sin_vector = sin.(0:0.1:2π)
const dl = DataLoader(reshape(sin_vector, 1, length(sin_vector), 1))

function setup_network(dl::DataLoader{T}) where T
    arch = Chain(Dense(1, 20, tanh), ResNetLayer(20, tanh), Dense(20, 1, identity))
    NeuralNetwork(arch, CPU(), T)
end

# tests checks if Adam with decay achieves a lower loss value than regular Adam and the two converge reasonably well
function train_network(; n_epochs=2048)
    nn₁ = setup_network(dl)
    nn₂ = setup_network(dl)

    o₁ = Optimizer(AdamOptimizer(), nn₁)
    o₂ = Optimizer(AdamOptimizerWithDecay(n_epochs), nn₂)

    batch = Batch(5, 1)
    loss = FeedForwardLoss()

    loss_array₁ = o₁(nn₁, dl, batch, n_epochs, loss)
    loss_array₂ = o₂(nn₂, dl, batch, n_epochs, loss)

    T = eltype(dl)
    @test loss_array₂[end] < loss_array₁[end] < T(1.6e-1)
end

@doc raw"""
`AdamOptimizerWithDecay` also has to work with manifold weights.

It is not one of `GeometricOptimizers`' own methods, so it needs to be routed onto GO's Adam cache
explicitly; without that every weight -- including the `StiefelManifold` ones -- falls through to the
Euclidean state, whose zero element is a `StiefelLieAlgHorMatrix` and not a manifold point.
"""
function train_manifold_network(; n_epochs = 32)
    arch = Chain(StiefelLayer(1, 20), Dense(20, 20, tanh), Dense(20, 1, identity))
    nn = NeuralNetwork(arch, CPU(), eltype(dl))

    o = Optimizer(AdamOptimizerWithDecay(n_epochs), nn)
    loss_array = o(nn, dl, Batch(5, 1), n_epochs, FeedForwardLoss(); show_progress = false)

    @test all(isfinite, loss_array)
    @test loss_array[end] < loss_array[1]
    # the Stiefel weight has to still be on the manifold
    Y = params(nn).L1.weight
    @test Y' * Y ≈ I
end

train_network()
train_manifold_network()
