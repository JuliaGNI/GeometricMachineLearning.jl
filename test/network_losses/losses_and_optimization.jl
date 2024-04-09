using GeometricMachineLearning
using Test 
import Random 

Random.seed!(123)

const sin_vector = sin.(0:0.01:2π)
const dl = DataLoader(reshape(sin_vector, 1, length(sin_vector)))

function setup_network(dl::DataLoader{T}) where T
    arch = GSympNet(dl)
    NeuralNetwork(dl, CPU(), T)
end

function train_network()
    nn = setup_network(dl)
    loss = FeedForwardLoss()
end