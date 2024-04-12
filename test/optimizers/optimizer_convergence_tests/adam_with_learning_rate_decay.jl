
using GeometricMachineLearning
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
    loss = GeometricMachineLearning.FeedForwardLoss()

    loss_array₁ = o₁(nn₁, dl, batch, n_epochs, loss)
    loss_array₂ = o₂(nn₂, dl, batch, n_epochs, loss)

    T = eltype(dl)
    @test loss_array₂[end] < loss_array₁[end] < T(1.6e-1)
end

train_network()
