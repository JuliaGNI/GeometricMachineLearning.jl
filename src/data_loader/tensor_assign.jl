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


@kernel function assign_output_estimate_kernel!(output_estimate::AbstractArray{T, 3}, full_output::AbstractArray{T,3}, seq_length, prediction_window) where T
    i,j,k = @index(Global, NTuple)
    output_estimate[i,j,k] = full_output[i,seq_length-prediction_window+j,k]
end

@doc raw"""
The function `assign_output_estimate` is closely related to the transformer. It takes the last `prediction_window` columns of the output and uses them for the final prediction.
i.e.
```math
\mathbb{R}^{N\times\mathtt{pw}}\to\mathbb{R}^{N\times\mathtt{pw}}, 
\begin{bmatrix} 
    z^{(1)}_1               & \cdots & z^{(T)}_1 \\ 
    \cdots                  & \cdots & \cdots    \\ 
    z^{(1)}_n               & \cdots & z^{(T})_n
    \end{bmatrix} \mapsto 
    \begin{bmatrix} 
    z^{(T - \mathtt{pw})}_1 & \cdots      & z^{(T)}_1 \\ 
    \cdots                  & \cdots      & \cdots \\ 
    z^{(T - \mathtt{pw})}_n & \cdots      & z^{(T})_n\end{bmatrix}     
``` 
"""
function assign_output_estimate(full_output::AbstractArray{T, 3}, prediction_window::Int) where T
    sys_dim, seq_length, batch_size = size(full_output)
    backend = KernelAbstractions.get_backend(full_output)
    output_estimate = KernelAbstractions.allocate(backend, T, sys_dim, prediction_window, batch_size)
    assign_output_estimate! = assign_output_estimate_kernel!(KernelAbstractions.get_backend(full_output))
    assign_output_estimate!(output_estimate, full_output, seq_length, prediction_window, ndrange=size(output_estimate))
    output_estimate
end

#=
"""
This function draws random time steps and parameters and based on these assign the batch and the output.

For ODE DataLoader: 
(i) batch input tensor to be written on
(ii) batch output tensor to be written on 
(iii) data tensor 

For MNIST DataLoader:
(i) batch input tensor to be written on
(ii) batch output tensor to be written on 
(iii) data tensor 
(iv) target tensor
"""
function draw_batch!(batch::AbstractArray{T, 3}, output::AbstractArray{T, 3}, data::AbstractArray{T, 3}) where T
    batch_size = size(batch, 3)
    n_params = size(data, 2)
    seq_length = size(batch, 2)
    prediction_window = size(output, 2)
    n_time_steps = size(data, 3)
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
    assign_output!(output, data, params, time_steps, seq_length, ndrange=size(output))
end

function draw_batch!(batch::AT, output::BT, data::AT, target::BT) where {T, T2, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T2, 3}}
    batch_size = size(batch, 3)
    n_params = size(data, 2)
    backend = KernelAbstractions.get_backend(batch)
    params = KernelAbstractions.allocate(backend, T, batch_size)
    rand!(Random.default_rng(), params)
    params = Int.(ceil.(n_params*params))
    assign_batch! = assign_batch_kernel!(backend)
    # for mnist this is fixed!
    time_steps = KernelAbstractions.ones(backend, T2, batch_size)
    assign_batch!(batch, data, params, time_steps, ndrange=size(batch))
    assign_batch!(output, target, params, time_steps, ndrange=size(output))
end
=#

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

function ChainRulesCore.rrule(::typeof(assign_output_estimate), full_output::AbstractArray{T, 3}, prediction_window) where T
    seq_length = size(full_output, 2)
    output_estimate = assign_output_estimate(full_output, prediction_window)
    function assign_output_estimate_pullback(output_diff)
        f̄ = NoTangent()
        batch_diff = @thunk augment_zeros(output_diff, seq_length)
        return f̄, batch_diff, NoTangent()
    end
    return output_estimate, assign_output_estimate_pullback
end     

augment_zeros(output_diff::Thunk, seq_length) = Thunk(() -> augment_zeros(unthunk(output_diff), seq_length))