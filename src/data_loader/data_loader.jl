"""
Data Loader is a struct that creates an instance based on a tensor (or different input format) and is designed to make training convenient.

Implemented: 
If the data loader is called with a single tensor, a batch_size and an output_size, then the batch is drawn randomly in the relevant range and the output is assigned accordingly.

The fields of the struct are the following: 
    - data: The input data with axes (i) system dimension, (ii) number of parameters and (iii) number of time steps.
    - batch: A tensor in which the current batch is stored. 
    - target_tensor: A tensor in which the current target is stored. 
    - output: The tensor from which the output is drawn - this may be of type Nothing if the constructor is only called with one tensor.
    - sys_dim: The ``dimension'' of the system, i.e. what is taken as input by a regular neural network. 
    - seq_length: The length batches should have. 
    - batch_size: 
    - output_size: The size of the second axis of the output tensor (prediction_window, output_size=1 in most cases)
    - n_params: The number of parameters that are present in the data set (length of second axis). 
    - n_time_steps: Number of time steps (length of third axis).

For drawing the batch, the sampling is done over n_params and n_time_steps (here seq_length and output_size are also taken into account).

TODO: Implement DataLoader that works well with GeometricEnsembles etc. 
"""
struct DataLoader{T, AT<:AbstractArray{T}, TT<:Union{AbstractArray, Nothing}, OT<:AbstractArray}
    data::AT
    batch::AT
    target_tensor::TT
    output::OT
    sys_dim::Integer
    seq_length::Integer 
    batch_size::Integer
    output_size::Integer 
    n_params::Integer 
    n_time_steps::Integer 
end

function DataLoader(data::AbstractArray{T, 3}, seq_length=10, batch_size=32, output_size=1) where T
    @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of parameters and (iii) number of time steps."
    sys_dim,n_params,n_time_steps = size(data)
    backend = KernelAbstractions.get_backend(data)
    batch = KernelAbstractions.allocate(backend, T, sys_dim, seq_length, batch_size)
    output = KernelAbstractions.allocate(backend, T, sys_dim, output_size, batch_size)
    draw_batch!(batch, output, data, seq_length, batch_size, output_size, n_params, n_time_steps)
    DataLoader{T, typeof(data), Nothing, typeof(output)}(data, batch, nothing, output, sys_dim, seq_length, batch_size, output_size, n_params, n_time_steps)
end

# T and T1 are not the same because T1 is of Integer type
function DataLoader(data::AbstractArray{T, 3}, target::AbstractVector{T1}; batch_size=32, patch_length=7) where {T, T1} 
    @info "You provided a tensor and a vector as input. This will be treated as a classification problem (MNIST). Tensor axes: (i) & (ii) image axes and (iii) batch dimesnion."
    im_dim₁, im_dim₂, batch_size = size(data)
    @assert length(target) == batch_size 
    number_of_patches = (im_dim₁÷patch_length)*(im_dim₂÷patch_length) 
    n_params = length(target)
    target = onehotbatch(target)
    data_preprocessed = split_and_flatten(data, patch_length, number_of_patches)
    backend = KernelAbstractions.get_backend(data)
    batch_input = KernelAbstractions.allocate(backend, T, patch_length^2, number_of_patches, batch_size)
    batch_output = KernelAbstractions.allocate(backend, T1, 10, 1, batch_size)
    DataLoader{T, typeof(data_preprocessed), typeof(target), typeof(batch_output)}(
        data_preprocessed, batch_input, target, batch_output, patch_length^2, number_of_patches, batch_size, 1, n_params, number_of_patches
        )
end

function redraw_batch(dl::DataLoader{T, AT, BT}) where {T, AT<:AbstractArray{T}, BT<:AbstractArray{<:Integer}}
    draw_batch!(dl.batch, dl.output, dl.data, dl.target_tensor, dl.batch_size, dl.n_params)
end

function redraw_batch(dl::DataLoader{T, AT, Nothing}) where {T, AT<:AbstractArray{T}}
    draw_batch!(dl.batch, dl.output, dl.data, dl.target, dl.seq_length, dl.batch_size, dl.n_params)
end

function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T}) where T
    batch_output = model(dl.batch, ps)
    output_estimate = assign_output_estimate(batch_output, dl.output_size)
    norm(dl.output - output_estimate)/T(sqrt(dl.batch_size))/T(sqrt(dl.output_size))
end