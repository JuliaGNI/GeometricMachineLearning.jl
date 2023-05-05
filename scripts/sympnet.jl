# Importe module
using GeometricMachineLearning

#Import Data
include("data_problem.jl")

data_q =  0
data_p = 0

# number of inputs/dimension of system
const ninput = 2
# layer dimension/width
const ld = 5
# hidden layers
const ln = 1
# activation function
const act = tanh
# number of training runs
const nruns = 1000


# Optimiser
opt = MomentumOptimizer(1e-2, 0.5)

# Creation of the architecture
sympnet = GSympNet(ninput, opt; width=ld, nhidden=ln, activation=act)

# create Lux network
nn = NeuralNetwork(sympnet, LuxBackend())

# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, data_q, data_p; ntraining = nruns)

#Plots
 