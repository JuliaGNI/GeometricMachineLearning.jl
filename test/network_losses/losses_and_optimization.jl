using GeometricMachineLearning
using GeometricMachineLearning: ResNetLayer
using Test 
import Random 

Random.seed!(123)

const sin_vector = sin.(0:0.01:2π)
const dl = DataLoader(reshape(sin_vector, 1, length(sin_vector), 1))

function setup_network(dl::DataLoader{T}) where T
    arch = Chain(Dense(dl.input_dim, 5, tanh), ResNetLayer(5, tanh), Dense(5, 1, identity))
    NeuralNetwork(arch, CPU(), T)
end

function train_network(; n_epochs=10)
    nn = setup_network(dl)
    loss = FeedForwardLoss()

    o = Optimizer(AdamOptimizer(), nn)
    batch = Batch(5, 1)
    loss_array = o(nn, dl, batch, n_epochs, loss)
    T = eltype(dl)
    @test loss_array[end] / loss_array[1] < T(0.1)
end

train_network()