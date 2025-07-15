# Importe module
using GeometricMachineLearning

# Generate Data
include("data_problem.jl")

nameproblem = :pendulum

H , n_dim = dict_problem_H[nameproblem]

Data = get_multiple_trajectory_structure(nameproblem; n_trajectory = 1, n_points = 1000, timestep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

get_Data = Dict(
    :Δt => Data -> Data.Δt,
    :nb_trajectory => Data -> Data.nb_trajectory,
    :length_trajectory => (Data,i) -> Data.data[Symbol("Trajectory_"*string(i))][:len],
    :q => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][1],
    :p => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][2],
)
data = DataTrajectory(Data, get_Data)


# Creation of the neural network

ld = 10             # layer dimension/width
ln = 2              # hidden layers
ninput = 2*n_dim    # number of inputs/dimension of system
act = tanh          # activation function

arch = GSympNet(ninput, width=ld, nhidden=ln, activation=act)

T = Float64
backend = CPU()

sympnet = NeuralNetwork(arch, backend, T)


# number of training runs
nruns = 1000
method = BasicSympNet() 
opt = MomentumOptimizer()

training_parameters = TrainingParameters(nruns, method, opt; batch_size  = batch_size)

# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, opt, data; ntraining = nruns, showprogress = true, timer = true)



q0 = [0.5]
p0 = [0.7]

#predictions
q_learned, p_learned = Iterate_Sympnet(nn, q0, p0; n_points = 100)

using Plots
using LaTeXStrings

nameproblem = :pendulum

data_q, data_p = get_phase_space_data(nameproblem, q0, p0, (0,2pi),0.1)

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

savefig("sympnet_pendulum.png")
