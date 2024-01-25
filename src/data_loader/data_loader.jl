description(::Val{:DataLoader}) = raw"""
Data Loader is a struct that creates an instance based on a tensor (or different input format) and is designed to make training convenient. 

The fields of the struct are the following: 
- `data`: The input data with axes (i) system dimension, (ii) number of parameters and (iii) number of time steps.
- `output`: The tensor that contains the output (supervised learning) - this may be of type Nothing if the constructor is only called with one tensor (unsupervised learning).
- `input_dim`: The *dimension* of the system, i.e. what is taken as input by a regular neural network.
- `input_time_steps`: The length of the entire time series of the data
- `n_params`: The number of parameters that are present in the data set (length of third axis)
- `output_dim`: The dimension of the output tensor (first axis). 
- `output_time_steps`: The size of the second axis of the output tensor (also called `prediction_window`, `output_time_steps=1` in most cases)

If for the output we have a tensor whose second axis has length 1, we still store it as a tensor and not a matrix for consistency. 
"""

"""
$(description(Val(:DataLoader)))
"""
struct DataLoader{T, AT<:Union{NamedTuple, AbstractArray{T}}, OT<:Union{AbstractArray, Nothing}, TimeSteps}
    input::AT
    output::OT
    input_dim::Int
    input_time_steps::Union{Int, Nothing}
    n_params::Union{Int, Nothing} 
    output_dim::Union{Int, Nothing}
    output_time_steps::Union{Int, Nothing}
end

struct TimeSteps end 
struct RegularData end

function DataLoader(data::AbstractArray{T, 3}) where T
    @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of time steps and (iii) number of params."
    input_dim, input_time_steps, n_params = size(data)
    DataLoader{T, typeof(data), Nothing, TimeSteps}(data, nothing, input_dim, input_time_steps, n_params, nothing, nothing)
end

description(::Val{:data_loader_constructor_matrix}) = raw"""
The constructor for the data loader, when called on a matrix, also takes an optional argument `autoencoder`. If set to true than the data loader assumes we are dealing with an *autoencoder problem* and the field `n_params` in the `DataLoader` object will be set to the number of columns of our input matrix. 
If `autoencoder=false`, then the field `input_time_steps` of the `DataLoader` object will be set to the *number of columns minus 1*. This is because in this case the data are used to train a neural network integrator and we need to leave at least one time step after the last one free in order to have something that we can compare the prediction to. 
So we have that for an input of form ``(z^{(0)}, \ldots, z^{(T)})`` `input_time_steps` is ``T``. 
"""

"""
$(description(Val(:data_loader_constructor_matrix)))
"""
function DataLoader(data::AbstractMatrix{T}; autoencoder=true) where T 
    @info "You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters."
    
    if autoencoder ==false
        input_dim, time_steps = size(data)
        return DataLoader{T, typeof(data), Nothing, TimeSteps}(data, nothing, input_dim, time_steps-1, nothing, nothing, nothing)
    elseif autoencoder ==true 
        input_dim, n_params = size(data)
        return DataLoader{T, typeof(data), Nothing, RegularData}(data, nothing, input_dim, nothing, n_params, nothing, nothing)
    end
end

# T and T1 are not the same because T1 is of Integer type
function DataLoader(data::AbstractArray{T, 3}, target::AbstractVector{T1}; patch_length=7) where {T, T1} 
    @info "You provided a tensor and a vector as input. This will be treated as a classification problem (MNIST). Tensor axes: (i) & (ii) image axes and (iii) parameter dimension."
    im_dim₁, im_dim₂, n_params = size(data)
    @assert length(target) == n_params 
    number_of_patches = (im_dim₁ ÷ patch_length) * (im_dim₂ ÷ patch_length) 
    target = onehotbatch(target)
    data_preprocessed = split_and_flatten(data, patch_length=patch_length, number_of_patches=number_of_patches)
    DataLoader{T, typeof(data_preprocessed), typeof(target), RegularData}(
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
        return DataLoader{T, typeof(data), Nothing, TimeSteps}(data, nothing, dim2 * 2, time_steps - 1, nothing, nothing, nothing)
    elseif autoencoder == true
        dim2, n_params = size(data.q)
        return DataLoader{T, typeof(data), Nothing, RegularData}(data, nothing, dim2 * 2, nothing, n_params, nothing, nothing)
    end
end

function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}}; output_time_steps=1) where {T, AT<:AbstractArray{T, 3}}
    @info "You have provided a NamedTuple with keys q and p; the data are tensors. This is interpreted as *symplectic data*."
    
    dim2, time_steps, n_params = size(data.q)
    DataLoader{T, typeof(data), Nothing, TimeSteps}(data, nothing, dim2 * 2, time_steps - 1, n_params, nothing, output_time_steps)
end

@doc raw"""
Computes the accuracy (as opposed to the loss) of a neural network classifier. 

It takes as input:
- `model::Chain`:
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