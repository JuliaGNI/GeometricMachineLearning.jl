#import module
using GeometricMachineLearning

# Import data
include("data_problem.jl")

data, target = get_LNN_data(:pendulum)

# layer dimension/width
const ld = 5

# hidden layers
const ln = 1

# number of inputs/dimension of system
const ninput = 2

# learning rate
const η = .001

# number of training runs
const nruns = 1000

# create HNN
lnn = LagrangianNeuralNetwork(ninput; nhidden = ln, width = ld)

# create Lux network
nn = NeuralNetwork(lnn, LuxBackend())


# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, data, target; ntraining = nruns, learning_rate = η)

