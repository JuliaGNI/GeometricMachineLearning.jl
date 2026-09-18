# A `Float32` network's training history used to come back `Float64`: `(o::Optimizer)(...)` built
# `loss_array` with an untyped `zeros(n_epochs)`, no matter what `eltype(dl)` was. This asserts the
# fix, and that `optimize_for_one_epoch!`'s own accumulator agrees with it: the loss the optimizer
# returns must be in the data's element type, not promoted to `Float64` along the way.

using GeometricMachineLearning
using Test
import Random

Random.seed!(11)

function test_training_history_eltype(T::DataType)
    nn = NeuralNetwork(GSympNet(4), CPU(), T)
    dl = DataLoader(rand(T, 4, 64); autoencoder = false, suppress_info = true)
    o = Optimizer(GradientOptimizer(), nn)

    history = o(nn, dl, Batch(8), 2, FeedForwardLoss(); show_progress = false)

    @test eltype(history) == T
    @test history isa Vector{T}
end

test_training_history_eltype(Float32)
test_training_history_eltype(Float64)
