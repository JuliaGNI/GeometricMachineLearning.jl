using GeometricMachineLearning, KernelAbstractions, LinearAlgebra, ProgressMeter, Zygote
using ChainRulesCore
using CUDA
using Random

include("generate_data.jl")

backend = CPU()
T = Float32

data_raw = generate_data()
dim, n_params, n_time_steps = size(data_raw)

data = KernelAbstractions.allocate(backend, T, size(data_raw))
copyto!(data, data_raw)

# I probably don't even have to declare all of the below as constant
const transformer_dim = 20
const num_heads = 4
const seq_length = 50
const n_epochs = 500
const batch_size = 128
const prediction_window = 5
include("auxiliary_functions.jl")

#=
model = Chain(  MultiHeadAttention(dim,2,Stiefel=true),
                Gradient(dim,5*dim,tanh,change_q=true),
		        Gradient(dim,5*dim,tanh,change_q=false),
		        MultiHeadAttention(dim,2,Stiefel=true),
                Gradient(dim,5*dim,tanh,change_q=false),
		        Gradient(dim,5*dim,tanh,change_q=true), 
		        MultiHeadAttention(dim,2,Stiefel=true),
                Gradient(dim,5*dim,tanh,change_q=true),
		        Gradient(dim,5*dim,tanh,change_q=false),
                Gradient(dim,5*dim,identity,change_q=true),
		        Gradient(dim,5*dim,identity,change_q=false))
=#

model = Chain(  Dense(dim, transformer_dim, tanh),
              MultiHeadAttention(transformer_dim, num_heads),
              ResNet(transformer_dim, tanh),
              MultiHeadAttention(transformer_dim, num_heads),
              ResNet(transformer_dim, tanh),
              Dense(transformer_dim, dim, identity)
              )

loss(ps) = loss(model, ps)
ps = initialparameters(backend, T, model)
o = Optimizer(AdamOptimizer(), ps)

n_training_steps_per_epoch = Int(ceil(n_time_steps/batch_size))
n_training_steps = n_epochs*n_training_steps_per_epoch

progress_object = Progress(n_training_steps; enabled=true)

for t in 1:n_training_steps
    draw_batch!(batch, output, seq_length, prediction_window)
    loss_val, pullback = Zygote.pullback(loss, ps)
    dx = pullback(1)[1]
    optimization_step!(o, model, ps, dx)
    ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss,loss_val)])
end
