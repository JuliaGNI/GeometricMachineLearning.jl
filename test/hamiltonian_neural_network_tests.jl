using GeometricMachineLearning
using Test
import Random 

Random.seed!(1234)

# input dimension
const dimin = 2

# layer width
const width = 5

# hidden layers
const nhidden = 1

# activation function
const act = tanh

# initialize HNN architecture
arch = HamiltonianNeuralNetwork(dimin; width=width, nhidden=nhidden, activation=act)

# create Lux network
nn = NeuralNetwork(arch, CPU(), Float64)

# create model for comparison
model = Chain(Dense(dimin, width, act),
              Dense(width, width, act),
              Linear(width, 1; use_bias=false))

# initialize Lux params and state
# TODO: Need to seed RNG in order to compare params and state with nn
# params, state = Lux.setup(Random.default_rng(), model)

@test nn.model == model
# @test nn.params == params
# @test nn.state == state
