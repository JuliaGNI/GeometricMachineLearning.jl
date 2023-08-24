"""
A few data loader/tensor assignment functions that were developed during Cemracs. 

They assign batched tensors during training based on a tensor with three axes: (i) system dimension, (ii) parameter dependence (including initial conditions), (iii) time step.

All of this uses KernelAbstractions and should work with any GPU supported by Julia.
"""

"""
Takes as input a ``batch tensor'' (to which the data are assigned), the whole data tensor and two vectors ``params'' and ``time_steps'' that include the specific parameters and time steps we want to assign. 

Note that this assigns sequential data! For e.g. being processed by a transformer.
"""
@kernel function assign_batch_kernel!(batch::AbstractArray{T, 3}, data::AbstractArray{T, 3}, params, time_steps) where T
    i,j,k = @index(Global, NTuple)
    time_step = time_steps[k]
    param = params[k]
    batch[i,j,k] = data[i,param,time_step-1+j]
end

"""
This should be used together with assign_batch_kernel!. It assigns the corresponding output (i.e. target).
"""
@kernel function assign_output_kernel!(output::AbstractArray{T, 3}, data::AbstractArray{T,3}, params, time_steps, seq_length::Integer) where T 
    i,j,k = @index(Global, NTuple)
    time_step = time_steps[k]
    param = params[k]
    output[i,j,k] = data[i,param,time_step+seq_length+j-1]
end


@kernel function assign_output_estimate_kernel!(output_estimate::AbstractArray{T, 3}, batch::AbstractArray{T,3}, seq_length, prediction_window) where T
    i,j,k = @index(Global, NTuple)
    output_estimate[i,j,k] = batch[i,seq_length-prediction_window+j,k]
end
"""
Closely related to the transformer. It takes the last prediction_window columns of the output and uses is for the final prediction.
"""
function assign_output_estimate(batch::AbstractArray{T, 3}, prediction_window) where T
    sys_dim, seq_length, batch_size = size(batch)
    backend = KernelAbstractions.get_backend(batch)
    output_estimate = KernelAbstractions.allocate(backend, T, sys_dim, prediction_window, batch_size)
    assign_output_estimate! = assign_output_estimate_kernel!(KernelAbstractions.get_backend(batch))
    assign_output_estimate!(output_estimate, batch, seq_length, prediction_window, ndrange=size(output_estimate))
    output_estimate
end

"""
This function draws random time steps and parameters and based on these assign the batch and the output.
"""
function draw_batch!(batch::AbstractArray{T, 3}, output::AbstractArray{T, 3}, data::AbstractArray{T, 3}, seq_length, prediction_window, n_params, n_time_steps) where T
    backend = KernelAbstractions.get_backend(batch)
    params = KernelAbstractions.allocate(backend, T, batch_size)
	time_steps = KernelAbstractions.allocate(backend, T, batch_size)
	rand!(Random.default_rng(), params)
	rand!(Random.default_rng(), time_steps)
	params = Int.(ceil.(n_params*params))
    time_steps = Int.(ceil.((n_time_steps-seq_length+1-prediction_window)*time_steps)) 
    assign_batch! = assign_batch_kernel!(backend)
    assign_output! = assign_output_kernel!(backend)
    assign_batch!(batch, data, params, time_steps, ndrange=size(batch))
    assign_output!(output, data, params, time_steps, ndrange=size(output))
end

function loss(model, ps, batch::AbstractArray{T, 3}, output::AbstractArray{T}) where T 
    sys_dim, prediction_window, batch_size = size(output)
    seq_length = size(batch, 2)
    batch_output = model(batch, ps)
    output_estimate = assign_output_estimate(batch_output, seq_length, prediction_window)
    norm(output - output_estimate)/T(sqrt(batch_size))/T(sqrt(prediction_window))
end

"""
Used for differentiating assign_output_estimate (this appears in the loss). 
"""
@kernel function augment_zeros_kernel!(zero_tensor::AbstractArray{T, 3}, output_diff::AbstractArray{T, 3}, seq_length, prediction_window) where T 
    i,j,k = @index(Global, NTuple)
    zero_tensor[i,seq_length-prediction_window+j,k] = output_diff[i,j,k]
end
function augment_zeros(output_diff::AbstractArray{T, 3}, seq_length) where T
    sys_dim, prediction_window, batch_size = size(output_diff)
    backend = KernelAbstractions.get_backend(output_diff)
    dim, prediction_window, batch_size = size(output_diff)
    zero_tensor = KernelAbstractions.zeros(backend, T, sys_dim, seq_length, batch_size)
    augment_zeros! = augment_zeros_kernel!(KernelAbstractions.get_backend(output_diff))
    augment_zeros!(zero_tensor, output_diff, seq_length, prediction_window, ndrange=size(output_diff))
    zero_tensor
end

function ChainRulesCore.rrule(::typeof(assign_output_estimate), batch::AbstractArray{T, 3}, prediction_window) where T
    seq_length = size(batch, 2)
    output_estimate = assign_output_estimate(batch, prediction_window)
    function assign_output_estimate_pullback(output_diff)
        f̄ = NoTangent()
        batch_diff = @thunk augment_zeros(output_diff, seq_length)
        return f̄, batch_diff, NoTangent()
    end
    return output_estimate, assign_output_estimate_pullback
end     

augment_zeros(output_diff::Thunk, seq_length) = Thunk(() -> augment_zeros(unthunk(output_diff), seq_length))