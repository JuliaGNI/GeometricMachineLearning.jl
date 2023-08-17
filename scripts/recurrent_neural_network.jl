using GeometricMachineLearning
using CUDA

# Generation of the data

include("transformer_coupled_oscillator/generate_data.jl")


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



# Paramters of the training 

batch_size = 500
nruns = 500

#training_parameters = TrainingParameters(nruns, method, opt, batch_size)

# Training of the NeuralNetwork

