# using Profile
using GeometricMachineLearning


# this contains the functions for generating the training data
include("data.jl")

# this contains the functions for generating the plots
# include("plots.jl")

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
hnn = HamiltonianNeuralNetwork(ninput; nhidden = ln, width = ld)

# create Lux network
nn = NeuralNetwork(hnn, LuxBackend())

# get data set
data, target = get_data_set()

# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, data, target; ntraining = nruns, learning_rate = η)

#time training (after warmup)
# total_loss = train!(hnn, data, target; ntraining = nruns, learning_rate = η)
# @time total_loss = train!(hnn, data, target; ntraining = nruns, learning_rate = η)

#profile training
#run with julia --track-allocation=user hnn.jl
# Profile.clear()
# Profile.clear_malloc_data()
# @profile total_loss = train!(hnn, data, target; ntraining = nruns, learning_rate = η)

#learned Hamiltonian & vector field
H_est(τ) = sum(apply(τ, hnn))
# dH_est(τ) = vectorfield(τ, hnn)

#plot results
# plot_network(H, H_est, total_loss; filename="hnn_simple.png")
