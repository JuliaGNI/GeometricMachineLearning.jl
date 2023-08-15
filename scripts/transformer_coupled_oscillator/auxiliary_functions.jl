using KernelAbstractions, ChainRulesCore, LinearAlgebra

batch = KernelAbstractions.allocate(backend, T, sys_dim, seq_length, batch_size)
output = KernelAbstractions.allocate(backend, T, sys_dim, prediction_window, batch_size)
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
@kernel function assign_output_kernel!(output::AbstractArray{T, 3}, data::AbstractArray{T,3}, params, time_steps) where T 
    i,j,k = @index(Global, NTuple)
    time_step = time_steps[k]
    param = params[k]
    output[i,j,k] = data[i,param,time_step+seq_length+j-1]
end
assign_output! = assign_output_kernel!(backend)

# this kernel assigns the output estimate
@kernel function assign_output_estimate_kernel!(output_estimate::AbstractArray{T, 3}, batch::AbstractArray{T,3}, seq_length, prediction_window) where T
    i,j,k = @index(Global, NTuple)
    output_estimate[i,j,k] = batch[i,seq_length-prediction_window+j,k]
end
assign_output_estimate! = assign_output_estimate_kernel!(backend)
function assign_output_estimate(batch::AbstractArray{T, 3}, seq_length, prediction_window) where T
    output_estimate = KernelAbstractions.allocate(backend, T, sys_dim, prediction_window, batch_size)
    assign_output_estimate!(output_estimate, batch, seq_length, prediction_window, ndrange=size(output_estimate))
    output_estimate
end

# draw batch (for one training step)
function draw_batch!(batch::AbstractArray{T, 3}, output::AbstractArray{T, 3}, seq_length, prediction_window) where T
	params = KernelAbstractions.allocate(backend, T, batch_size)
	time_steps = KernelAbstractions.allocate(backend, T, batch_size)
	rand!(Random.default_rng(), params)
	rand!(Random.default_rng(), time_steps)
	params = Int.(ceil.(n_params*params))
    time_steps = Int.(ceil.((n_time_steps-seq_length+1-prediction_window)*time_steps)) 
    assign_batch!(batch, data, params, time_steps, ndrange=size(batch))
    assign_output!(output, data, params, time_steps, ndrange=size(output))
end

function loss(model, ps, batch::AbstractArray{T, 3}, output::AbstractArray{T}, seq_length) where T 
    prediction_window = size(output, 2)
    batch_output = model(batch, ps)
    output_estimate = assign_output_estimate(batch_output, seq_length, prediction_window)
    norm(output - output_estimate)/sqrt(batch_size)/sqrt(prediction_window)
end
loss(model, ps) = loss(model, ps, batch, output, seq_length)

@kernel function augment_zeros_kernel!(zero_tensor::AbstractArray{T, 3}, output_diff::AbstractArray{T, 3}, seq_length, prediction_window) where T 
    i,j,k = @index(Global, NTuple)
    zero_tensor[i,seq_length-prediction_window+j,k] = output_diff[i,j,k]
end
augment_zeros! = augment_zeros_kernel!(backend)
function augment_zeros(output_diff::AbstractArray{T, 3}, seq_length) where T
    dim, prediction_window, batch_size = size(output_diff)
    zero_tensor = KernelAbstractions.zeros(backend, T, sys_dim, seq_length, batch_size)
    augment_zeros!(zero_tensor, output_diff, seq_length, prediction_window, ndrange=size(output_diff))
    zero_tensor
end

function ChainRulesCore.rrule(::typeof(assign_output_estimate), batch::AbstractArray{T, 3}, seq_length, prediction_window) where T
    output_estimate = assign_output_estimate(batch, seq_length, prediction_window)
    function assign_output_estimate_pullback(output_diff::AbstractArray)
        f̄ = NoTangent()
        batch_diff = @thunk augment_zeros(output_diff, seq_length)
        return f̄, batch_diff, NoTangent(), NoTangent()
    end
    return output_estimate, assign_output_estimate_pullback
end     

# using ChainRulesTestUtils
# draw_batch!(batch, output)
# test_rrule(assign_output_estimate, batch)