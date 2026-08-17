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

using CairoMakie
using LaTeXStrings

nameproblem = :pendulum

data_q, data_p = get_phase_space_data(nameproblem, q0, p0, (0,2pi),0.1)

plt = Figure(size = (1000, 800))

ax_qp = Axis(plt[1, 1]; title = "G-SympNet prediction for the simple pendulum", titlesize = 15,
             xlabel = L"q", ylabel = L"p", xlabelsize = 18, ylabelsize = 18,
             xticklabelsize = 10, yticklabelsize = 10)
lines!(ax_qp, data_q[:,1], data_p[:,1]; label = "Training data.", linewidth = 3)
lines!(ax_qp, q_learned[:,1], p_learned[:,1]; label = "Learned trajectory.", linewidth = 3)
axislegend(ax_qp; position = :lb, nbanks = 2, labelsize = 15)

ax_loss = Axis(plt[2, 1]; title = "Total loss during the training", titlesize = 15,
               xlabel = L"n_{training}", ylabel = L"Loss", xlabelsize = 18, ylabelsize = 18,
               xticklabelsize = 10, yticklabelsize = 10)
lines!(ax_loss, total_loss; label = "Loss.", linewidth = 3)
axislegend(ax_loss; position = :lb, nbanks = 2, labelsize = 15)

# the loss occupies the bottom 40% of the figure
rowsize!(plt.layout, 2, Relative(0.4))

CairoMakie.save("sympnet_pendulum.png", plt)
