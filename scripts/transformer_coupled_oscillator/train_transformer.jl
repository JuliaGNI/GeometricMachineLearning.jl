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

const transformer_dim = 20
const num_heads = 4
const seq_length = 50
const n_epochs = 500
const batch_size = 128
const prediction_window = 5

model = Chain(  Dense(dim, transformer_dim, tanh),
              MultiHeadAttention(transformer_dim, num_heads),
              ResNet(transformer_dim, tanh),
              MultiHeadAttention(transformer_dim, num_heads),
              ResNet(transformer_dim, tanh),
              Dense(transformer_dim, dim, identity)
              )

ps = initialparameters(backend, T, model)
o = Optimizer(AdamOptimizer(), ps)

batch = KernelAbstractions.allocate(backend, T, dim, seq_length, batch_size)
output = prediction_window == 1 ? KernelAbstractions.allocate(backend, T, dim, batch_size) : KernelAbstractions.allocate(backend, T, dim, prediction_window, batch_size)
#output_estimate = prediction_window == 1 ? KernelAbstractions.allocate(backend, T, dim, batch_size) : KernelAbstractions.allocate(backend, T, dim, prediction_window, batch_size)

# this kernel draws a batch based on arrays of parameters and time_steps
@kernel function assign_batch_kernel!(batch::AbstractArray{T, 3}, data::AbstractArray{T, 3}, params, time_steps) where T
    i,j,k = @index(Global, NTuple)
    time_step = time_steps[k]
    param = params[k]
    batch[i,j,k] = data[i,param,time_step-1+j]
end
assign_batch! = assign_batch_kernel!(backend)

# this kernel assigns the output based on the batch
@kernel function assign_output_kernel!(output::AbstractMatrix{T}, data::AbstractArray{T,3}, params, time_steps) where T 
    i,j = @index(Global, NTuple)
    time_step = time_steps[j]
    param = params[j]
    output[i,j] = data[i,param,time_step+seq_length]
end
@kernel function assign_output_kernel!(output::AbstractArray{T, 3}, data::AbstractArray{T,3}, params, time_steps) where T 
    i,j,k = @index(Global, NTuple)
    time_step = time_steps[k]
    param = params[k]
    output[i,j,k] = data[i,param,time_step+seq_length+j-1]
end
assign_output! = assign_output_kernel!(backend)

# this kernel assigns the output estimate
@kernel function assign_output_estimate_kernel!(output_estimate::AbstractMatrix{T}, batch::AbstractArray{T,3}) where T
    i,j= @index(Global, NTuple)
    output_estimate[i,j] = batch[i,seq_length,j]
end
@kernel function assign_output_estimate_kernel!(output_estimate::AbstractArray{T, 3}, batch::AbstractArray{T,3}) where T
    i,j,k = @index(Global, NTuple)
    output_estimate[i,j,k] = batch[i,seq_length-prediction_window+j,k]
end
assign_output_estimate! = assign_output_estimate_kernel!(backend)
function assign_output_estimate(batch::AbstractArray{T, 3}) where T
    output_estimate = prediction_window == 1 ? KernelAbstractions.allocate(backend, T, dim, batch_size) : KernelAbstractions.allocate(backend, T, dim, prediction_window, batch_size)
    assign_output_estimate!(output_estimate, batch, ndrange=size(output_estimate))
    output_estimate
end

# draw batch (for one training step)
function draw_batch!(batch::AbstractArray{T, 3}, output::AbstractArray{T}) where T
	params = KernelAbstractions.allocate(backend, T, batch_size)
	time_steps = KernelAbstractions.allocate(backend, T, batch_size)
	rand!(Random.default_rng(), params)
	rand!(Random.default_rng(), time_steps)
	params = Int.(ceil.(n_params*params))
    time_steps = Int.(ceil.((n_time_steps-seq_length+1-prediction_window)*time_steps)) 
    assign_batch!(batch, data, params, time_steps, ndrange=size(batch))
    assign_output!(output, data, params, time_steps, ndrange=size(output))
end

function loss(ps, batch::AbstractArray{T, 3}, output::AbstractArray{T}) where T 
    batch_output = model(batch, ps)
    output_estimate = assign_output_estimate(batch_output)
    norm(output - output_estimate)/sqrt(batch_size)/sqrt(prediction_window)
end
loss(ps) = loss(ps, batch, output)

@kernel function augment_zeros_kernel!(zero_tensor::AbstractArray{T, 3}, output_diff::AbstractMatrix{T}) where T
    i,j = @index(Global, NTuple)
    zero_tensor[i,seq_length,j] = output_diff[i,j]
end
@kernel function augment_zeros_kernel!(zero_tensor::AbstractArray{T, 3}, output_diff::AbstractArray{T, 3}) where T 
    i,j,k = @index(Global, NTuple)
    zero_tensor[i,seq_length-prediction_window+j,k] = output_diff[i,j,k]
end
augment_zeros! = augment_zeros_kernel!(backend)
function augment_zeros(output_diff::AbstractArray{T}) where T
    zero_tensor = KernelAbstractions.zeros(backend, T, dim, seq_length, batch_size)
    augment_zeros!(zero_tensor, output_diff, ndrange=size(output_diff))
    zero_tensor
end

function ChainRulesCore.rrule(::typeof(assign_output_estimate), batch::AbstractArray{T, 3}) where T
    output_estimate = assign_output_estimate(batch)
    function assign_output_estimate_pullback(output_diff::AbstractArray)
        f̄ = NoTangent()
        batch_diff = @thunk augment_zeros(output_diff)
        return f̄, batch_diff
    end
    return output_estimate, assign_output_estimate_pullback
end     

# using ChainRulesTestUtils
# draw_batch!(batch, output)
# test_rrule(assign_output_estimate, batch)
n_training_steps_per_epoch = Int(ceil(n_time_steps/batch_size))
n_training_steps = n_epochs*n_training_steps_per_epoch

progress_object = Progress(n_training_steps; enabled=true)

for t in 1:n_training_steps
    draw_batch!(batch, output)
    loss_val, pullback = Zygote.pullback(loss, ps)
    dx = pullback(1)[1]
    optimization_step!(o, model, ps, dx)
    ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss,loss_val)])
end
