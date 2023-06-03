# Importe module
using GeometricMachineLearning

#Import Data
#include("pendulum.jl")
#data_q, data_p =  pendulum_data()
#plt = plot(data_q, data_p, label="Training data.")

include("data_problem.jl")

nameproblem = :pendulum
q0 = [0.2]
p0 = [0.8]


#nameproblem = :Hénon_Heiles
#q0 = [0.3,-0.3]
#p0 = [0.3,0.15]

_, n_dim = dict_problem_H[nameproblem]

#data_q, data_p = get_phase_space_data(nameproblem, q0, p0, (0,100pi),0.01)
n_trajectory = 100
data_q, data_p = get_phase_space_multiple_trajectoy(nameproblem, singlematrix = true, n_trajectory = n_trajectory, n_points = 30, tstep = 0.1)


#=
plt = plot(data_q[1][:,1], data_q[1][:,2], label="Training data.",linewidth = 3)
for j in 2:n_trajectory
    plot!(data_q[j][:,1], data_q[j][:,2], label="Training data.",linewidth = 3)
end
return plt
=#


# number of inputs/dimension of system
const ninput = 2*n_dim
# layer dimension/width
const ld = 10
# hidden layers
const ln = 2
# activation function
const act = tanh
# number of training runs
const nruns = 10000
# batch size
const batch_size = 10


# Optimiser
opt = MomentumOptimizer(1e-2, 0.5)

# Creation of the architecture
#sympnet = GSympNet(ninput, width=ld, nhidden=ln, activation=act)
sympnet = GSympNet(ninput, width=ld, nhidden=ln, activation=act)

# create Lux network
nn = NeuralNetwork(sympnet, LuxBackend())

# perform training (returns array that contains the total loss for each training step)
#=total_loss = train!(nn, opt, data_q, data_p; ntraining = nruns, batch_size=batch_size)

#predictions
q_learned, p_learned = Iterate_Sympnet(nn, q0, p0; n_points = size(data_q,1))




#Plots
using LaTeXStrings

data_q, data_p = get_phase_space_data(nameproblem, q0, p0, (0,6pi),0.1)

plt_qp = plot(data_q[:,1], data_p[:,1], label="Training data.",linewidth = 3,mk=*)
plot!(plt_qp, q_learned[:,1], p_learned[:,1], label="Learned trajectory.", linewidth = 3, guidefontsize=18, tickfontsize=10, size=(1000,800), legendfontsize=15, titlefontsize=15)
title!("G-SympNet prediction for the simple pendulum")
xlabel!(L"q")
ylabel!(L"p")
plot!(legend=:outerbottom,legendcolumns=2)

plt_loss = plot(total_loss,linewidth = 3, label="Loss.", guidefontsize=18, tickfontsize=10, size=(1000,800), legendfontsize=15, titlefontsize=15)
title!("Total loss during the training")
xlabel!(L"n_{training}")
ylabel!(L"Loss")
plot!(legend=:outerbottom,legendcolumns=2)

l = @layout [
    grid(1,1)
    b{0.4h}
]

plt = plot(plt_qp, plt_loss, layout = l)


savefig("sympnet_henon_heiles.png")


=#