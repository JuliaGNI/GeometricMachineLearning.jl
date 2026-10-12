# A `Float32` network's training history on `Float32` data is `Float32`: `train!` gives the history
# the type of the loss value, and the epoch's accumulator keeps that type, so nothing is promoted to
# `Float64` along the way. `test/data_loader/train.jl` checks a network and data of two element
# types.

using GeometricMachineLearning
using Test
import Random

Random.seed!(11)

function test_training_history_eltype(T::DataType)
    nn = NeuralNetwork(GSympNet(4), CPU(), T)
    dl = DataLoader(rand(T, 4, 64); autoencoder = false, suppress_info = true)
    o = TrainingOptimizer(nn.params; algorithm = GradientMethod())

    history = train!(
        nn, o, dl, Batch(8), 2; loss = FeedForwardLoss(), show_progress = false)

    @test eltype(history) == T
    @test history isa Vector{T}
end

test_training_history_eltype(Float32)
test_training_history_eltype(Float64)
