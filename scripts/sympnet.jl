# Importe module
using GeometricMachineLearning
using Lux

#Import Data
#include("pendulum.jl")
#data_q, data_p =  pendulum_data()
#plt = plot(data_q, data_p, label="Training data.")

include("data_problem.jl")

nameproblem = "pendulum"
q0 = [0.2]
p0 = [0.8]


#nameproblem = "Hénon_Heiles"
#q0 = [0.3,-0.3]
#p0 = [0.3,0.15]

_, n_dim = dict_problem[nameproblem]

data_q, data_p = get_phase_space_data(nameproblem, q0, p0, (0,6pi),0.01)


#plt = plot(data_q[:,1], data_p[:,1], label="Training data.")

plt = plot(data_q[:,1], data_p[:,1], label="Training data.")



# number of inputs/dimension of system
const ninput = 2*n_dim
# layer dimension/width
const ld = 10
# hidden layers
const ln = 2
# activation function
const act = tanh
# number of training runs
const nruns = 1000


# Optimiser
opt = MomentumOptimizer(1e-2, 0.5)


# Creation of the architecture
sympnet = GSympNet(ninput, width=ld, nhidden=ln, activation=act)


# create Lux network
nn = NeuralNetwork(sympnet, LuxBackend())


# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, opt, data_q, data_p; ntraining = nruns)


#predictions
q_learned, p_learned = Iterate_Sympnet(nn, q0, p0; n_points = size(data_q,1))


#Plots

plot(plt, q_learned[:,1], p_learned[:,1], label="Learned trajectory.")
savefig("sympnet_pendulum_Architecture_test.png")
