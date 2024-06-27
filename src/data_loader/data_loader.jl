description(::Val{:DataLoader}) = raw"""
    DataLoader(data)

Make an instance based on a data set.

This is designed such to make training convenient.

# Constructor 

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

# Fields of `DataLoader`

The fields of the `DataLoader` struct are the following: 
- `input`: The input data with axes (i) system dimension, (ii) number of time steps and (iii) number of parameters.
- `output`: The tensor that contains the output (supervised learning) - this may be of type `Nothing` if the constructor is only called with one tensor (unsupervised learning).
- `input_dim`: The *dimension* of the system, i.e. what is taken as input by a regular neural network.
- `input_time_steps`: The length of the entire time series (length of the second axis).
- `n_params`: The number of parameters that are present in the data set (length of third axis)
- `output_dim`: The dimension of the output tensor (first axis). If `output` is of type `Nothing`, then this is also of type `Nothing`.
- `output_time_steps`: The size of the second axis of the output tensor. If `output` is of type `Nothing`, then this is also of type `Nothing`.

# Implementation

Even though `DataLoader` can be called with inputs of various forms, internally it always stores tensors with three axes.

```jldoctest
using GeometricMachineLearning

data = [1 2 3; 4 5 6]
dl = DataLoader(data)
dl.input

# output

[ Info: You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters.
2×1×3 Array{Int64, 3}:
[:, :, 1] =
 1
 4

[:, :, 2] =
 2
 5

[:, :, 3] =
 3
 6
```
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

"""
    DataLoader(data::AbstractMatrix)

Make an instance of `DataLoader` based on a matrix.

# Arguments 

This has an addition keyword argument `autoencoder`, which is set to `true` by default.
"""
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

function data_matrices_from_geometric_solution(solution::GeometricSolution{T, <:Number, NT}) where {T <: Number, DT <: DataSeries{T}, NT<:NamedTuple{(:q, :p), Tuple{DT, DT}}}
    sys_dim, input_time_steps = length(solution.s.q[0]), length(solution.t)
    data = (q = zeros(T, sys_dim, input_time_steps), p = zeros(T, sys_dim, input_time_steps))

    for dim in 1:sys_dim 
        data.q[dim, :] = solution.q[:, dim]
        data.p[dim, :] = solution.p[:, dim]
    end

    data
end

function DataLoader(solution::GeometricSolution{T, <:Number, NT}; kwargs...) where {T <: Number, DT <: DataSeries{T}, NT<:NamedTuple{(:q, :p), Tuple{DT, DT}}}
    data = data_matrices_from_geometric_solution(solution)

    DataLoader(data; kwargs...)
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

"""
Constructor for `EnsembleSolution` form package `GeometricSolutions` with fields `q` and `p`.
"""
function DataLoader(ensemble_solution::EnsembleSolution{T, T1, Vector{ST}}) where {T, T1, DT <: DataSeries{T}, ST <: GeometricSolution{T, T1, NamedTuple{(:q, :p), Tuple{DT, DT}}}}
    sys_dim, input_time_steps, n_params = length(ensemble_solution.s[1].q[0]), length(ensemble_solution.t), length(ensemble_solution.s)

    data = (q = zeros(T, sys_dim, input_time_steps, n_params), p = zeros(T, sys_dim, input_time_steps, n_params))

    for (solution, i) in zip(ensemble_solution.s, axes(ensemble_solution.s, 1))
        for dim in 1:sys_dim 
            data.q[dim, :, i] = solution.q[:, dim]
            data.p[dim, :, i] = solution.p[:, dim]
        end 
    end

    DataLoader(data)
end

function map_to_new_backend(input::AbstractArray{T}, backend::KernelAbstractions.Backend) where T
    input₂ = KernelAbstractions.allocate(backend, T, size(input)...)
    KernelAbstractions.copyto!(backend, input₂, input)
    input₂
end

function map_to_new_backend(input::QPT{T}, backend::KernelAbstractions.Backend) where T
    input₂ = (q = KernelAbstractions.allocate(backend, T, size(input.q)...), p = KernelAbstractions.allocate(backend, T, size(input.p)...))
    KernelAbstractions.copyto!(backend, input₂.q, input.q)
    KernelAbstractions.copyto!(backend, input₂.p, input.p)
    input₂
end

function map_to_type(input::QPT, T::DataType)
    (q = T.(input.q), p = T.(input.p))
end

function map_to_type(input::AbstractArray, T::DataType)
    T.(input)
end

function DataLoader(dl::DataLoader{T1, <:QPTOAT, Nothing}, backend::KernelAbstractions.Backend, T::DataType=T1) where T1
    input = 
        if T==T1
            dl.input
        else
            map_to_new_type(dl.input, T)
        end

    DataLoader(map_to_new_backend(dl.input, backend))
end

function reshape_to_matrix(dl::DataLoader{<:Number, <:QPT, Nothing, :RegularData})
    (q = reshape(dl.input.q, dl.input_dim ÷ 2, dl.n_params), p = reshape(dl.input.p, dl.input_dim ÷ 2, dl.n_params))
end

function reshape_to_matrix(dl::DataLoader{<:Number, <:AbstractArray, Nothing, :RegularData})
    reshape(dl.input, dl.input_dim, dl.n_params)
end

function DataLoader(dl::DataLoader{T1, <: QPTOAT, Nothing, :RegularData}, 
    backend::KernelAbstractions.Backend=KernelAbstractions.get_backend(dl),
    T::DataType=T1;
    autoencoder::Bool=true) where T1

    if autoencoder == true
        input = 
        if T == T1
            dl.input
        else
            map_to_type(dl.input, T)
        end 

        new_input = map_to_new_backend(input, backend)
        DataLoader{T, typeof(new_input), Nothing, :RegularData}(
            new_input,
            nothing,
            dl.input_dim,
            dl.input_time_steps,
            dl.n_params,
            nothing,
            nothing)
    elseif autoencoder == false
        matrix_data = reshape_to_matrix(dl)

        matrix_data_new_type = 
            if T == T1
                matrix_data
            else
                map_to_type(matrix_data, T)
            end

        DataLoader(map_to_new_backend(matrix_data_new_type, backend); autoencoder=false)
    end
end

@doc raw"""
    accuracy(model, ps, dl)

Compute the accuracy of a neural network classifier. 
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

"""
    accuracy(nn, dl)

Compute the accuracy of a neural network classifier.
"""
accuracy(nn::NeuralNetwork, dl::DataLoader) = accuracy(nn.model, nn.params, dl)

Base.eltype(::DataLoader{T}) where T = T

KernelAbstractions.get_backend(dl::DataLoader) = KernelAbstractions.get_backend(dl.input)
function KernelAbstractions.get_backend(dl::DataLoader{T, <:QPT{T}}) where T
    KernelAbstractions.get_backend(dl.input.q)
end