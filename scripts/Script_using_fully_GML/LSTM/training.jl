using GeometricMachineLearning
using JLD2
using KernelAbstractions
using LinearAlgebra

# Generation of the data
backend = CPU()
T = Float32

file = jldopen("data", "r")
data_raw = file["tensor"]
sys_dim, n_params, n_time_steps = size(data_raw)
generated_data = KernelAbstractions.allocate(backend, T, size(data_raw))
copyto!(generated_data, data_raw)

get_Data = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> size(Data)[2],
    :length_trajectory => (Data,i) -> size(Data)[3],
    :Δt => Data -> 0.4,
    :q => (Data,i,n) -> [Data[1,i,n], Data[2,i,n]],
    :p => (Data,i,n) -> [Data[3,i,n], Data[4,i,n]]
)
data = TrainingData(generated_data, get_Data)

# Creation of the architecture

dimin = 4
dimout = 4
ssize = (1, 6)

arch = LSTMNeuralNetwork(dimin, dimout, ssize)

# Creation of the NeuralNetwork

rnn = NeuralNetwork(arch, backend, T)

# Creation of the Optimizer

opt = AdamOptimizer()

# Creation of the Loss Function

function loss(rnn, data, batch = ((1,1),(1,2)), params = rnn.params)
    s = 0
    for (i,j) in batch
        x = [[get_data(data, :q, i, y)...,get_data(data, :p, i, y)...] for y in j:(j+5)]
        y = [get_data(data, :q, i, j+6)...,get_data(data, :p, i, j+6)...]
        s += norm(y - rnn(x, params))
    end
    s
end

# Paramters of the training 

batch_size = (1,7,7)
nruns = 500

training_parameters = TrainingParameters(nruns, loss, opt; batch_size  = batch_size)

# Training of the NeuralNetwork

train!(rnn, data, training_parameters; showprogress = true, timer = true)



jldsave("nn_model", model=rnn, params = rnn.params, seq_length = 6)
