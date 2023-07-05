# using Profile
using GeometricMachineLearning

# this contains the functions for generating the training data
include("../data_problem.jl")


function HNN(integrator::TrainingIntegrator{<:HnnTrainingIntegrator}, data::AbstractTrainingData, nameproblem::Symbol = :pendulum, opt =  MomentumOptimizer(1e-3,0.5))
    
    _, n_dim = dict_problem_H[nameproblem]

    # layer dimension/width
    ld = 1

    # hidden layers
    ln = 0

    # number of inputs/dimension of system
    ninput = 2*n_dim

    # number of training runs
    nruns = 10

    # create HNN
    hnn = HamiltonianNeuralNetwork(ninput; nhidden = ln, width = ld)

    # create Lux network
    nn = NeuralNetwork(hnn, LuxBackend())

    # perform training (returns array that contains the total loss for each training step)
    total_loss = 0#train!(nn, data, opt, integrator; ntraining = nruns, showprogress = true)

    return nn, total_loss
end



Data= get_HNN_data(:pendulum)
Get_Data = Dict(
    :shape => SampledData,
    :nb_points => Data -> length(Data[1]),
    :q => (Data,n) -> Data[1][n][1],
    :p => (Data,n) -> Data[1][n][2],
    :q̇ => (Data,n) -> Data[2][n][1],
    :ṗ => (Data,n) -> Data[2][n][2],
)
data = TrainingData(Data, Get_Data)


nn, total_loss = HNN(ExactHnn(), data,  :pendulum, MomentumOptimizer())
index_batch = get_batch(data, 10)

loss_gradient(nn, type(ExactHnn()), data, index_batch, nn.params)





#include("../plots.jl")

#H,_ = dict_problem_H[:pendulum]

#plot_hnn(H, nn, total_loss; filename="hnn_pendulum.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)
