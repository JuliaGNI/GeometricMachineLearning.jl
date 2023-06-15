# Importe module
using GeometricMachineLearning

#Import Data
#include("pendulum.jl")
#data_q, data_p =  pendulum_data()
#plt = plot(data_q, data_p, label="Training data.")

include("data_problem.jl")

function SYMPNET(integrator::SympNetIntegrator, data::Training_data, nameproblem::Symbol = :pendulum, opt =  MomentumOptimizer(1e-3,0.5))
    
    _, n_dim = dict_problem_H[nameproblem]

    # layer dimension/width
    ld = 10

    # hidden layers
    ln = 2

    # number of inputs/dimension of system
    ninput = 2*n_dim

    # number of training runs
    nruns = 3

    # activation function
    act = tanh

    # create SympNet
    sympnet = GSympNet(ninput, width=ld, nhidden=ln, activation=act)

    # create Lux network
    nn = NeuralNetwork(sympnet, LuxBackend())

    # perform training (returns array that contains the total loss for each training step)
    total_loss = train!(nn, opt, data; ntraining = nruns, hti = integrator)

    return nn, total_loss
end


Data = get_multiple_trajectory_structure(:pendulum; n_trajectory = 2, n_points = 3, tstep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

Get_Data = Dict(
    :Δt => Data -> Data.Δt,
    :nb_trajectory => Data -> Data.nb_trajectory,
    :length_trajectory => (Data,i) -> Data.data[Symbol("Trajectory_"*string(i))][:len],
    :q => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][1],
    :p => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][2],
)
data2 = data_trajectory(Data, Get_Data)

SYMPNET(BaseIntegrator(), data2, :pendulum, MomentumOptimizer())




#_, n_dim = dict_problem_H[nameproblem]

#data_q, data_p = get_phase_space_data(nameproblem, q0, p0, (0,100pi),0.01)
#n_trajectory = 100
#data_q, data_p = get_phase_space_multiple_trajectoy(nameproblem, singlematrix = true, n_trajectory = n_trajectory, n_points = 30, tstep = 0.1)


#=
plt = plot(data_q[1][:,1], data_q[1][:,2], label="Training data.",linewidth = 3)
for j in 2:n_trajectory
    plot!(data_q[j][:,1], data_q[j][:,2], label="Training data.",linewidth = 3)
end
return plt
=#









#=Plots
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