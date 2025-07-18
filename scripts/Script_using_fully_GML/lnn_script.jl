#import module
using GeometricMachineLearning

# this contains the functions for generating the training data
include("../data_problem.jl")


function LNN(integrator::TrainingIntegrator{<:LnnTrainingIntegrator}, data::AbstractTrainingData, nameproblem::Symbol = :pendulum, opt =  MomentumOptimizer(1e-3,0.5))
    
    _, n_dim = dict_problem_L[nameproblem]

    # layer dimension/width
    ld = 5

    # hidden layers
    ln = 3

    # number of inputs/dimension of system
    ninput = 2*n_dim

    # number of training runs
    nruns = 3

    # create lNN
    lnn = LagrangianNeuralNetwork(ninput; nhidden = ln, width = ld)

    # create Lux network
    nn = NeuralNetwork(lnn, LuxBackend())

    # perform training (returns array that contains the total loss for each training step)
    total_loss = train!(nn, opt, data; ntraining = nruns, ti = integrator, showprogress = true)

    return nn, total_loss
end


#=
Data = get_multiple_trajectory_structure(:pendulum; n_trajectory = 20, n_points = 1000, timestep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

Get_Data = Dict(
    :Δt => Data -> Data.Δt,
    :nb_trajectory => Data -> Data.nb_trajectory,
    :length_trajectory => (Data,i) -> Data.data[Symbol("Trajectory_"*string(i))][:len],
    :q => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][1],
)
data = DataTrajectory(Data, Get_Data)

nn, total_loss = LNN(VariationalMidPointIntegrator(), data, :pendulum, MomentumOptimizer())


# plot results
include("../plots.jl")  
L, n_dim = dict_problem_L[:pendulum]
plot_hnn(L, nn, total_loss; filename="lnn_pendulum.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)
=#
