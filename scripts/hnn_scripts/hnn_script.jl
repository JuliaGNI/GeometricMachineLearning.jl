# using Profile
using GeometricMachineLearning

# this contains the functions for generating the training data
include("../data_problem.jl")


function HNN(integrator::TrainingIntegrator{<:HnnTrainingIntegrator}, data::AbstractTrainingData, nameproblem::Symbol = :pendulum, opt =  MomentumOptimizer(1e-3,0.5))
    
    _, n_dim = dict_problem_H[nameproblem]

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

    # create Lux network
    nn = NeuralNetwork(hnn, LuxBackend())

    # perform training (returns array that contains the total loss for each training step)
    total_loss = train!(nn, opt, data; ntraining = nruns, ti = integrator, showprogress = false)

    return nn, total_loss
end



Data,Target = get_HNN_data(:pendulum)

Get_Data = Dict(
    :nb_points => Data -> length(Data),
    :q => (Data,n) -> Data[n][1],
    :p => (Data,n) -> Data[n][2]
)
pdata = DataSampled(Data, Get_Data)

Get_Target = Dict(
    :q̇ => (Target,n) -> Target[n][1],
    :ṗ => (Target,n) -> Target[n][2],
)

data = DataTarget(pdata, Target, Get_Target)

nn, total_loss = HNN(ExactHnn(), data,  :pendulum, MomentumOptimizer())

include("../plots.jl")

H,_ = dict_problem_H[:pendulum]

plot_hnn(H, nn, total_loss; filename="hnn_pendulum.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)
