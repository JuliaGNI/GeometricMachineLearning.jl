description(::Val{:DataLoader}) = raw"""
Data Loader is a struct that creates an instance based on a tensor (or different input format) and is designed to make training convenient. 

## Constructor 

The data loader can be called with various inputs:
- **A single vector**: If the data loader is called with a single vector (and no other arguments are given), then this is interpreted as an autoencoder problem, i.e. the second axis indicates parameter values and/or time steps and the system has a single degree of freedom (i.e. the system dimension is one).
- **A single matrix**: If the data loader is called with a single matrix (and no other arguments are given), then this is interpreted as an autoencoder problem, i.e. the first axis is assumed to indicate the degrees of freedom of the system and the second axis indicates parameter values and/or time steps. 
- **A single tensor**: If the data loader is called with a single tensor, then this is interpreted as an *integration problem* with the second axis indicating the time step and the third one indicating the parameters.
- **A tensor and a vector**: This is a special case (MNIST classification problem). For the MNIST problem for example the input are ``n_p`` matrices (first input argument) and ``n_p`` integers (second input argument).
- **A `NamedTuple` with fields `q` and `p`**: The `NamedTuple` contains (i) two matrices or (ii) two tensors. 
- **An `EnsembleSolution`**: The `EnsembleSolution` typically comes from `GeometricProblems`.

When we supply a single vector or a single matrix as input to `DataLoader` and further set `autoencoder = false` (keyword argument), then the data are stored as an *integration problem* and the second axis is assumed to indicate time steps.
"""

"""
$(description(Val(:DataLoader)))

## Fields of `DataLoader`

The fields of the `DataLoader` struct are the following: 
    - `input`: The input data with axes (i) system dimension, (ii) number of time steps and (iii) number of parameters.
    - `output`: The tensor that contains the output (supervised learning) - this may be of type `Nothing` if the constructor is only called with one tensor (unsupervised learning).
    - `input_dim`: The *dimension* of the system, i.e. what is taken as input by a regular neural network.
    - `input_time_steps`: The length of the entire time series (length of the second axis).
    - `n_params`: The number of parameters that are present in the data set (length of third axis)
    - `output_dim`: The dimension of the output tensor (first axis). If `output` is of type `Nothing`, then this is also of type `Nothing`.
    - `output_time_steps`: The size of the second axis of the output tensor. If `output` is of type `Nothing`, then this is also of type `Nothing`.

### The `input` and `output` fields of `DataLoader`

Even though the arguments to the Constructor may be vectors or matrices, internally `DataLoader` always stores tensors.
"""
struct DataLoader{T, AT<:Union{NamedTuple, AbstractArray{T}}, OT<:Union{AbstractArray, Nothing}, DataType}
    input::AT
    output::OT
    input_dim::Int
    input_time_steps::Int
    n_params::Int
    output_dim::Union{Int, Nothing}
    output_time_steps::Union{Int, Nothing}
end

function DataLoader(data::AbstractArray{T, 3}) where T
    @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of time steps and (iii) number of params."
    input_dim, input_time_steps, n_params = size(data)
    DataLoader{T, typeof(data), Nothing, :TimeSeries}(data, nothing, input_dim, input_time_steps, n_params, nothing, nothing)
end

function DataLoader(data::AbstractMatrix{T}; autoencoder=true) where T 
    @info "You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters."
    
    if autoencoder ==false
        input_dim, time_steps = size(data)
        reshaped_data = reshape(data, input_dim, time_steps, 1)
        return DataLoader{T, typeof(reshaped_data), Nothing, :TimeSeries}(reshaped_data, nothing, input_dim, time_steps, 1, nothing, nothing)
    elseif autoencoder ==true 
        input_dim, n_params = size(data)
        reshaped_data = reshape(data, input_dim, 1, n_params)
        return DataLoader{T, typeof(reshaped_data), Nothing, :RegularData}(reshaped_data, nothing, input_dim, 1, n_params, nothing, nothing)
    end
end

DataLoader(data::AbstractVector; autoencoder=true) = DataLoader(reshape(data, 1, length(data)); autoencoder = autoencoder)

# T and T1 are not the same because T1 is of Integer type
function DataLoader(data::AbstractArray{T, 3}, target::AbstractVector{T1}; patch_length=7) where {T, T1} 
    @info "You provided a tensor and a vector as input. This will be treated as a classification problem (MNIST). Tensor axes: (i) & (ii) image axes and (iii) parameter dimension."
    im_dim₁, im_dim₂, n_params = size(data)
    @assert length(target) == n_params 
    number_of_patches = (im_dim₁ ÷ patch_length) * (im_dim₂ ÷ patch_length) 
    target = onehotbatch(target)
    data_preprocessed = split_and_flatten(data, patch_length=patch_length, number_of_patches=number_of_patches)
    DataLoader{T, typeof(data_preprocessed), typeof(target), :RegularData}(
        data_preprocessed, target, patch_length^2, number_of_patches, n_params, 10, 1
        )
end

description(::Val{:data_loader_for_named_tuple}) =  raw"""
`DataLoader` can also be called with a `NamedTuple` that has `q` and `p` as keys.

In this case the field `input_dim` of `DataLoader` is interpreted as the sum of the ``q``- and ``p``-dimensions, i.e. if ``q`` and ``p`` both evolve on ``\mathbb{R}^n``, then `input_dim` is ``2n``.
"""

"""
$(description(Val(:DataLoader)))
"""
function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}}; autoencoder=false) where {T, AT<:AbstractMatrix{T}} 
    @info "You have provided a NamedTuple with keys q and p; the data are matrices. This is interpreted as *symplectic data*."
    
    if autoencoder == false
        dim2, time_steps = size(data.q)
        reshaped_data = (q = reshape(data.q, dim2, time_steps, 1), p = reshape(data.p, dim2, time_steps, 1))
        return DataLoader{T, typeof(reshaped_data), Nothing, :TimeSeries}(reshaped_data, nothing, dim2 * 2, time_steps, 1, nothing, nothing)
    elseif autoencoder == true
        dim2, n_params = size(data.q)
        reshaped_data = (q = reshape(data.q, dim2, 1, n_params), p = reshape(data.p, dim2, 1, n_params))
        return DataLoader{T, typeof(reshaped_data), Nothing, :RegularData}(reshaped_data, nothing, dim2 * 2, 1, n_params, nothing, nothing)
    end
end

function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}}) where {T, AT<:AbstractArray{T, 3}}
    @info "You have provided a NamedTuple with keys q and p; the data are tensors. This is interpreted as *symplectic data*."
    
    dim2, time_steps, n_params = size(data.q)
    DataLoader{T, typeof(data), Nothing, :TimeSeries}(data, nothing, dim2 * 2, time_steps, n_params, nothing, nothing)
end

DataLoader(data::NamedTuple{(:q, :p), Tuple{VT, VT}}) where {VT <: AbstractVector} = DataLoader((q = reshape(data.q, 1, length(data.q)), p = reshape(data.p, 1, length(data.p))))

"""
Constructor for `EnsembleSolution` form package `GeometricSolutions` with fields `q` and `p`.
"""
function DataLoader(ensemble_solution::EnsembleSolution{T, T1, Vector{ST}}) where {T, T1, DT, ST <: GeometricSolution{T, T1, NamedTuple{(:q, :p), Tuple{DT, DT}}}}

    sys_dim, input_time_steps, n_params = length(ensemble_solution.s[1].q[0]), length(ensemble_solution.t), length(ensemble_solution.s)

    data = (q = zeros(sys_dim, input_time_steps, n_params), p = zeros(sys_dim, input_time_steps, n_params))

    for (solution, i) in zip(ensemble_solution.s, axes(ensemble_solution.s, 1))
        for dim in 1:sys_dim 
            data.q[dim, :, i] = solution.q[:, dim]
            data.p[dim, :, i] = solution.p[:, dim]
        end 
    end

    DataLoader(data)
end

"""
Constructor for `EnsembleSolution` from package `GeometricSolutions` with field `q`.
"""
function DataLoader(ensemble_solution::EnsembleSolution{T, T1, Vector{ST}}) where {T, T1, DT, ST <: GeometricSolution{T, T1, NamedTuple{(:q, ), Tuple{DT}}}}

    sys_dim, input_time_steps, n_params = length(ensemble_solution.s[1].q[0]), length(ensemble_solution.t), length(ensemble_solution.s)

    data = zeros(sys_dim, input_time_steps, n_params)

    for (solution, i) in zip(ensemble_solution.s, axes(ensemble_solution.s, 1))
        for dim in 1:sys_dim 
            data[dim, :, i] = solution.q[:, dim]
        end 
    end

    DataLoader(data)
end

@doc raw"""
Computes the accuracy (as opposed to the loss) of a neural network classifier. 

It takes as input:
- `model::Chain`
- `ps`: parameters of the network
- `dl::DataLoader`
"""
function accuracy(model::Chain, ps::Tuple, dl::DataLoader{T, AT, BT}) where {T, T1<:Integer, AT<:AbstractArray{T}, BT<:AbstractArray{T1}}
    output_tensor = model(dl.input, ps)
    output_estimate = assign_output_estimate(output_tensor, dl.output_time_steps)
    backend = KernelAbstractions.get_backend(output_estimate)
    tensor_of_maximum_elements = KernelAbstractions.zeros(backend, T1, size(output_estimate)...)
    ind = argmax(output_estimate, dims=1)
    # get tensor of maximum elements
    tensor_of_maximum_elements[ind] .= T1(1)
    (size(dl.output, 3)-sum(abs.(dl.output - tensor_of_maximum_elements))/T1(2))/size(dl.output, 3)
end

accuracy(nn::NeuralNetwork, dl::DataLoader) = accuracy(nn.model, nn.params, dl)

Base.eltype(::DataLoader{T}) where T = T

KernelAbstractions.get_backend(dl::DataLoader) = KernelAbstractions.get_backend(dl.input)