# Importe module
using GeometricMachineLearning
using Lux

#Import Data
include("pendulum.jl")
data_q, data_p =  pendulum_data()
plt = plot(data_q, data_p, label="Training data.")


# number of inputs/dimension of system
const ninput = 2
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
sympnet = GSympNet(ninput, opt; width=ld, nhidden=ln, activation=act)

# create Lux network
nn = NeuralNetwork(sympnet, LuxBackend())



# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, data_q, data_p; ntraining = nruns)


#Plots

q_learned = zero(data_q)
p_learned = zero(data_p)
q_learned[0] = data_q[0]
p_learned[0] = data_p[0]

for i in 1:lastindex(data_q)
    q_learned[i], p_learned[i] = nn(q_learned[i-1], p_learned[i-1])
end 

# plot result and save figure to file
plot(plt, q_learned, p_learned, label="Learned trajectory.")
savefig("sympnet_pendulum_Architecture.png")
