# A `Float32` network's training history is `Float32`: `(o::Optimizer)(...)` builds `loss_array`
# typed to `eltype(dl)`. This asserts that, and that `optimize_for_one_epoch!`'s own accumulator
# agrees with it: the loss the optimizer returns is in the data's element type, not promoted to
# `Float64` along the way. See `CHANGELOG.md` for the fix this guards.

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
