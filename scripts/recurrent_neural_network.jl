using GeometricMachineLearning
using CUDA

# Generation of the data

include("transformer_coupled_oscillator/generate_data.jl")

generated_data = generate_data()

get_data = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> length(Data),
    :length_trajectory => (Data,i) -> length(Data[Symbol("Trajectory"*string(i))][1]),
    :Δt => Data -> 0.1,
    :q => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][1][n],
    :p => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][2][n],
)
data = TrainingData(generated_data, get_data)


# Creation of the architecture

dimin = 2
dimout = 2
size = (1, 6)

arch = RecurrentNeuralNetwork(dimin, dimout, size)

# Creation of the NeuralNetwork

backend = CPU()
T = Float32

rnn = NeuralNetwork(arch, backend, T)

# Creation of the Optimizer

opt = AdamOptimizer()

# Creation of the Loss Function

function loss(nn, data, batch = default, params = nn.params)


end

# Paramters of the training 

batch_size = 500
nruns = 500

#training_parameters = TrainingParameters(nruns, method, opt, batch_size)

# Training of the NeuralNetwork

