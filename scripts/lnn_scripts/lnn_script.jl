#import module
using GeometricMachineLearning

# this contains the functions for generating the training data
include("../data_problem.jl")


function LNN(integrator::Lnn_training_integrator, data::Training_data, nameproblem::Symbol = :pendulum, opt =  MomentumOptimizer(1e-3,0.5))
    
    _, n_dim = dict_problem_L[nameproblem]

    # layer dimension/width
    ld = 5

    # hidden layers
    ln = 3

    # number of inputs/dimension of system
    ninput = 2*n_dim

    # number of training runs
    nruns = 3

    # create HNN
    hnn = HamiltonianNeuralNetwork(ninput; nhidden = ln, width = ld)

    # create lNN
    lnn = LagrangianNeuralNetwork(ninput; nhidden = ln, width = ld)

    # create Lux network
    nn = NeuralNetwork(lnn, LuxBackend())

    # perform training (returns array that contains the total loss for each training step)
    total_loss = train!(nn, opt, data; ntraining = nruns, lti = integrator)

    return nn, total_loss
end


#=
Data, Target = get_LNN_data(nameproblem)

Get_Data = Dict(
    :nb_points => Data -> length(Data),
    :q => (Data,n) -> Data[n][1][1],
    :q̇ => (Data,n) -> Data[n][2][1]
)
pdata = data_sampled(Data, Get_Data)

Get_Target = Dict(
    :q̈ => (Target,n) -> Target[n][1],
)

data = dataTarget(pdata, Target, Get_Target)


Data = get_multiple_trajectory_structure_Lagrangian(nameproblem; n_trajectory = 2, n_points = 10)

Get_Data = Dict(
    :Δt => Data -> Data[1][1],
    :nb_trajectory => Data -> Data[2][1],
    :length_trajectory => (Data,i) -> Data[3][1],
    :q => (Data,i,n) -> Data[3+i][n],
)
data = data_trajectory(Data, Get_Data)

=#




# plot results
#include("plots.jl")
#plot_hnn(L, nn, total_loss; filename="lnn_pendulum.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)


