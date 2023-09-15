"""
Data Loader is a struct that creates an instance based on a tensor (or different input format) and is designed to make training convenient. \n
\n
Implemented: \n
If the data loader is called with a single tensor, a batch_size and an output_size, then the batch is drawn randomly in the relevant range and the output is assigned accordingly.\n
\n
The fields of the struct are the following: \n
\t   - data: The input data with axes (i) system dimension, (ii) number of parameters and (iii) number of time steps.\n
\t   - batch: A tensor in which the current batch is stored.\n 
\t   - target_tensor: A tensor in which the current target is stored.\n
\t   - output: The tensor from which the output is drawn - this may be of type Nothing if the constructor is only called with one tensor.\n
\t   - sys_dim: The ``dimension'' of the system, i.e. what is taken as input by a regular neural network.\n 
\t   - seq_length: The length batches should have.\n
\t   - batch_size:\n
\t   - output_size: The size of the second axis of the output tensor (prediction_window, output_size=1 in most cases)\n
\t   - n_params: The number of parameters that are present in the data set (length of second axis)\n
\t   - n_time_steps: Number of time steps (length of third axis)\n
\n
For drawing the batch, the sampling is done over n_params and n_time_steps (here seq_length and output_size are also taken into account).\n
\n
TODO: Implement DataLoader that works well with GeometricEnsembles etc.\n
"""
struct DataLoader{T, AT<:AbstractArray{T}, TT<:Union{AbstractArray, Nothing}, OT<:Union{AbstractArray, Nothing}}
    data::AT
    batch::AT
    target_tensor::TT
    output::OT
    sys_dim::Integer
    seq_length::Union{Integer, Nothing} 
    batch_size::Integer
    output_size::Union{Integer, Nothing} 
    n_params::Integer 
    n_time_steps::Union{Integer, Nothing}
end

function DataLoader(data::AbstractArray{T, 3}; seq_length=10, batch_size=32, output_size=1) where T
    @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of parameters and (iii) number of time steps."
    sys_dim,n_params,n_time_steps = size(data)
    backend = KernelAbstractions.get_backend(data)
    batch = KernelAbstractions.allocate(backend, T, sys_dim, seq_length, batch_size)
    output = KernelAbstractions.allocate(backend, T, sys_dim, output_size, batch_size)
    draw_batch!(batch, output, data)
    DataLoader{T, typeof(data), Nothing, typeof(output)}(data, batch, nothing, output, sys_dim, seq_length, batch_size, output_size, n_params, n_time_steps)
end

function DataLoader(data::AbstractMatrix{T}; batch_size=32) where T 
    @info "You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters."
    sys_dim, n_params = size(data)
    backend = KernelAbstractions.get_backend(data)
    batch = KernelAbstractions.allocate(backend, T, sys_dim, batch_size)
    draw_batch!(batch, data)
    DataLoader{T, typeof(data), Nothing, Nothing}(data, batch, nothing, nothing, sys_dim, nothing, batch_size, nothing, n_params, nothing)
end

# T and T1 are not the same because T1 is of Integer type
function DataLoader(data::AbstractArray{T, 3}, target::AbstractVector{T1}; batch_size=32, patch_length=7) where {T, T1} 
    @info "You provided a tensor and a vector as input. This will be treated as a classification problem (MNIST). Tensor axes: (i) & (ii) image axes and (iii) batch dimesnion."
    im_dim₁, im_dim₂, n_params = size(data)
    @assert length(target) == n_params 
    number_of_patches = (im_dim₁÷patch_length)*(im_dim₂÷patch_length) 
    target = onehotbatch(target)
    data_preprocessed = split_and_flatten(data, patch_length, number_of_patches)
    backend = KernelAbstractions.get_backend(data)
    batch_input = KernelAbstractions.allocate(backend, T, patch_length^2, number_of_patches, batch_size)
    batch_output = KernelAbstractions.allocate(backend, T1, 10, 1, batch_size)
    draw_batch!(batch_input, batch_output, data_preprocessed, target)
    DataLoader{T, typeof(data_preprocessed), typeof(target), typeof(batch_output)}(
        data_preprocessed, batch_input, target, batch_output, patch_length^2, number_of_patches, batch_size, 1, n_params, number_of_patches
        )
end

function redraw_batch!(dl::DataLoader{T, AT, BT}) where {T, AT<:AbstractArray{T}, BT<:AbstractArray{<:Integer}}
    draw_batch!(dl.batch, dl.output, dl.data, dl.target_tensor)
end

function redraw_batch!(dl::DataLoader{T, AT, Nothing}) where {T, AT<:AbstractArray{T}}
    draw_batch!(dl.batch, dl.output, dl.data)
end

function redraw_batch!(dl::DataLoader{T, AT, Nothing}) where {T, AT<:AbstractMatrix{T}}
    draw_batch!(dl.batch, dl.data)
end

function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T}) where T
    batch_output = model(dl.batch, ps)
    output_estimate = assign_output_estimate(batch_output, dl.output_size)
    norm(dl.output - output_estimate)/T(sqrt(dl.batch_size))/T(sqrt(dl.output_size))
end

function loss(model::Chain, ps::Tuple, dl::DataLoader{T}) where T 
    batch_output = model(dl.batch, ps)
    norm(batch_output - dl.batch)/norm(dl.batch)
end

function accuracy(model::Chain, ps::Tuple, dl::DataLoader{T, AT, BT}) where {T, T2<:Integer, AT<:AbstractArray{T}, BT<:AbstractArray{T2}}
    output_tensor = model(dl.batch, ps)
    output_estimate = assign_output_estimate(output_tensor, dl.output_size)
    backend = KernelAbstractions.get_backend(output_estimate)
    tensor_of_maximum_elements = KernelAbstractions.zeros(backend, T2, size(output_estimate)...)
    ind = argmax(output_estimate, dims=1)
    tensor_of_maximum_elements[ind] .= T2(1)
    (size(dl.output, 3)-sum(abs.(dl.output - tensor_of_maximum_elements))/T2(2))/size(dl.output, 3)
end
