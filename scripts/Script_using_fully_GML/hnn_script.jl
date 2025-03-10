# using Profile
using GeometricMachineLearning

# generation of data
include("data_problem.jl")

nameproblem = :pendulum

Data= get_HNN_data(nameproblem)
Get_Data = Dict(
    :shape => SampledData,
    :nb_points => Data -> length(Data[1]),
    :q => (Data,n) -> Data[1][n][1],
    :p => (Data,n) -> Data[1][n][2],
    :q̇ => (Data,n) -> Data[2][n][1],
    :ṗ => (Data,n) -> Data[2][n][2],
)
data = TrainingData(Data, Get_Data)

H, dim = dict_problem_H[nameproblem]


# Creation of the neural network

ninput = 2*dim
ln = 2
ld = 1

arch = HamiltonianArchitecture(ninput; nhidden = ln, width = ld)

hnn = NeuralNetwork(hnn, Float64)

# Parameters of the training

nruns = 1000
opt = MomentumOptimizer()
method = ExactHnn()
batch_size = 10

training_parameters = TrainingParameters(nruns, method, opt; batch_size  = batch_size)

# Training 

total_loss = train!(nn, data, opt, integrator; ntraining = nruns, showprogress = true)


# Plot 

include("plots.jl")

plot_hnn(H, hnn, total_loss; filename="hnn_pendulum.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)
