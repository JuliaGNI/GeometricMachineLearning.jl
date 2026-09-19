"""
A few data loader/tensor assignment functions that were developed during Cemracs. 

They assign batched tensors during training based on a tensor with three axes: (i) system dimension, (ii) parameter dependence (including initial conditions), (iii) time step.

All of this uses KernelAbstractions and should work with any GPU supported by Julia.
"""

# Assigning a batch and its target out of the full data tensor is `src/data_loader/batch.jl`'s job:
# `assign_input_from_vector_of_tuples_kernel!` and `assign_output_from_vector_of_tuples_kernel!`
# index by a vector of `(time_step, parameter)` tuples. What is here estimates the output instead.

@kernel function assign_output_estimate_kernel!(
        output_estimate::AbstractArray{T, 3}, full_output::AbstractArray{T, 3},
        seq_length, prediction_window) where {T}
    i, j, k = @index(Global, NTuple)
    output_estimate[i, j, k] = full_output[i, seq_length - prediction_window + j, k]
end

@doc raw"""
    assign_output_estimate(full_output, prediction_window)

Crop the output to get the correct number of output vectors.

The function `assign_output_estimate` is closely related to the [`Transformer`](@ref). 
It takes the last `prediction_window` columns of the output and uses them for the prediction.

i.e.
```math
\mathbb{R}^{N\times{}T}\to\mathbb{R}^{N\times\mathtt{pw}}, 
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

If `prediction_window` is equal to `sequence_length`, then this is not needed.
"""
function assign_output_estimate(full_output::AbstractArray{T, 3}, prediction_window::Int) where {T}
    sys_dim, seq_length, batch_size = size(full_output)
    backend = networkbackend(full_output)
    output_estimate = KernelAbstractions.allocate(
        backend, T, sys_dim, prediction_window, batch_size)
    assign_output_estimate! = assign_output_estimate_kernel!(networkbackend(full_output))
    assign_output_estimate!(output_estimate, full_output, seq_length,
        prediction_window, ndrange = size(output_estimate))
    output_estimate
end

# """
# Used for differentiating assign_output_estimate (this appears in the loss). 
# """
@kernel function augment_zeros_kernel!(
        zero_tensor::AbstractArray{T, 3}, output_diff::AbstractArray{T, 3},
        seq_length, prediction_window) where {T}
    i, j, k = @index(Global, NTuple)
    zero_tensor[i, seq_length - prediction_window + j, k] = output_diff[i, j, k]
end
function augment_zeros(output_diff::AbstractArray{T, 3}, seq_length) where {T}
    sys_dim, prediction_window, batch_size = size(output_diff)
    backend = networkbackend(output_diff)
    zero_tensor = KernelAbstractions.zeros(backend, T, sys_dim, seq_length, batch_size)
    augment_zeros! = augment_zeros_kernel!(networkbackend(output_diff))
    augment_zeros!(zero_tensor, output_diff, seq_length,
        prediction_window, ndrange = size(output_diff))
    zero_tensor
end

function ChainRulesCore.rrule(::typeof(assign_output_estimate),
        full_output::AbstractArray{T, 3}, prediction_window) where {T}
    seq_length = size(full_output, 2)
    output_estimate = assign_output_estimate(full_output, prediction_window)
    function assign_output_estimate_pullback(output_diff)
        f̄ = NoTangent()
        batch_diff = @thunk augment_zeros(unthunk(output_diff), seq_length)
        return f̄, batch_diff, NoTangent()
    end
    return output_estimate, assign_output_estimate_pullback
end
