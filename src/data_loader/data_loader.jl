@doc raw"""
    DataLoader(data)

Make an instance based on a data set.

This is designed to make training convenient.

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

"""
    DataLoader(data::AbstractArray{<:Number, 3})

Make an instance of DataLoader for data that are in tensor format.

# Arguments 

There are two optional keyword arguments:
- `autoencoder = false` and
- `suppress_info = false`.

By default the data are stored as `TimeSeries` type. If you want to train an [`AutoEncoder`](@ref) with your data call:

```julia
    DataLoader(data; autoencoder = true)
```

The default is equivalent to `autoencoder = false`.

By default we have:

```jldoctest
using GeometricMachineLearning

data = [ 1;  2;  3;; 
         4;  5;  6;;;
         7;  8;  9;; 
        10; 11; 12]

DataLoader(data)

# output

┌ Info: You have provided a tensor with three axes as input. They will be interpreted as
└  (i) system dimension, (ii) number of time steps and (iii) number of params.
DataLoader{Int64, Array{Int64, 3}, Nothing, :TimeSeries}([1 4; 2 5; 3 6;;; 7 10; 8 11; 9 12], nothing, 3, 2, 2, nothing, nothing)
```

But if we write

```jldoctest
using GeometricMachineLearning

data = [ 1;  2;  3;; 
         4;  5;  6;;;
         7;  8;  9;; 
        10; 11; 12]

DataLoader(data; suppress_info = true)

# output

DataLoader{Int64, Array{Int64, 3}, Nothing, :TimeSeries}([1 4; 2 5; 3 6;;; 7 10; 8 11; 9 12], nothing, 3, 2, 2, nothing, nothing)
```
the `@info` statement is not printed.
"""
function DataLoader(data::AbstractArray{T, 3}; autoencoder = false, suppress_info = false) where T
    if !suppress_info
        @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of time steps and (iii) number of params."
    end
    input_dim, input_time_steps, n_params = size(data)

    if autoencoder == false
        DataLoader{T, typeof(data), Nothing, :TimeSeries}(data, nothing, input_dim, input_time_steps, n_params, nothing, nothing)
    elseif autoencoder == true
        DataLoader{T, typeof(data), Nothing, :RegularData}(data, nothing, input_dim, input_time_steps, n_params, nothing, nothing)
    end
end

"""
    DataLoader(data::AbstractMatrix)

Make an instance of [`DataLoader`](@ref) based on a matrix.

# Arguments 

See [`DataLoader(::AbstractArray{<:Number, 3})`](@ref) for details.

# Implementation

Internally the data are reshaped to a tensor of shape `(size(data)..., 1)` to make for a consistent representation.
"""
function DataLoader(data::AbstractMatrix{T}; autoencoder=true, suppress_info = false) where T
    if !suppress_info
        @info "You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters."
    end

    if autoencoder == false
        input_dim, time_steps = size(data)
        reshaped_data = reshape(data, input_dim, time_steps, 1)
        return DataLoader{T, typeof(reshaped_data), Nothing, :TimeSeries}(reshaped_data, nothing, input_dim, time_steps, 1, nothing, nothing)
    elseif autoencoder == true 
        input_dim, n_params = size(data)
        reshaped_data = reshape(data, input_dim, 1, n_params)
        return DataLoader{T, typeof(reshaped_data), Nothing, :RegularData}(reshaped_data, nothing, input_dim, 1, n_params, nothing, nothing)
    end
end

"""
    DataLoader(data::AbstractVector)

Make an instance of `DataLoader` based on a vector.

# Extend Help

If the input to `DataLoader` is a vector, it is assumed that this vector represents one-dimensional time-series data and is therefore processed as:

```julia
    DataLoader(data::AbstractVector; autoencoder=true) = DataLoader(reshape(data, 1, length(data)); autoencoder = autoencoder)
```
"""
function DataLoader(data::AbstractVector; autoencoder=true, suppress_info = false)
    DataLoader(reshape(data, 1, length(data)); autoencoder = autoencoder, suppress_info = suppress_info)
end

function DataLoader(input::AbstractArray{T, 3}, output::AbstractArray{T, 3}; suppress_info = false) where T
    @assert size(input, 3) == size(output, 3)
    if !suppress_info
        @info "You have provided an input and an output."
    end

    DataLoader{T, typeof(input), typeof(output), :TimeSeries}(input, output, size(input, 1), size(input, 2), size(input, 3), size(output, 1), size(output, 2))
end

function DataLoader(input::AbstractMatrix{T}, output::AbstractMatrix{T}; suppress_info = false) where T
    DataLoader(reshape(input, size(input)..., 1), reshape(output, size(output)..., 1); suppress_info = suppress_info)
end

function DataLoader(input::AbstractVector{T}, output::AbstractVector{T}; suppress_info = false) where T
    DataLoader(reshape(input, 1, length(input)), reshape(output, 1, length(output)); suppress_info = suppress_info)
end

@doc raw"""
    DataLoader(data::AbstractArray{T, 3}, target::AbstractVector)

Make an instance of DataLoader for a classification problem. 

Target here is a vector of labels. This is tailored towards being used with the package [`MLDatasets.jl`](https://github.com/JuliaML/MLDatasets.jl).

# Arguments

There are two keyword arguments:
- `patch_length = 7`. This is the length of the patch in the ``x`` and the ``y`` direction;
- `suppress_info = false`.

For the example of the MNIST data set all images are of size ``49\times49``.
For `patch_length = 7` the image is therefore split into 16 ``7\times7`` patches [brantner2023generalizing](@cite).
"""
function DataLoader(data::AbstractArray{T, 3}, target::AbstractVector{T1}; 
                    patch_length=7,
                    suppress_info = false) where {T, T1}
    if !suppress_info
        @info "You provided a tensor and a vector as input. This will be treated as a classification problem (MNIST). Tensor axes: (i) & (ii) image axes and (iii) parameter dimension."
    end
    im_dim₁, im_dim₂, n_params = size(data)
    @assert length(target) == n_params 
    number_of_patches = (im_dim₁ ÷ patch_length) * (im_dim₂ ÷ patch_length) 
    target = onehotbatch(target)
    data_preprocessed = split_and_flatten(data, patch_length = patch_length, number_of_patches = number_of_patches)
    DataLoader{T, typeof(data_preprocessed), typeof(target), :TimeSeries}(
        data_preprocessed, target, patch_length^2, number_of_patches, n_params, 10, 1
        )
end

@doc raw"""
    DataLoader(data::QPT)

Make an instance of `DataLoader` based on ``(q, p)`` data.

# Implementation

In this case the field `input_dim` of `DataLoader` is interpreted as the sum of the ``q``- and ``p``-dimensions, i.e. if ``q`` and ``p`` both evolve on ``\mathbb{R}^n``, then `input_dim` is ``2n``.

Apart from this the input is treated similarly as if it were an `Array`, i.e. everything is converted to tensors internally.
See e.g. [`DataLoader{::AbstractArray{<:Number, 3}}`](@ref).
"""
function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}}; 
                    autoencoder=false, 
                    suppress_info = false) where {T, AT<:AbstractMatrix{T}}
    if !suppress_info
        @info "You have provided a NamedTuple with keys q and p; the data are matrices. This is interpreted as *symplectic data*."
    end

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

function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}}; 
                    autoencoder = false,
                    suppress_info = false) where {T, AT<:AbstractArray{T, 3}}
    if !suppress_info
        @info "You have provided a NamedTuple with keys q and p; the data are tensors. This is interpreted as *symplectic data*."
    end
    
    dim2, time_steps, n_params = size(data.q)

    if autoencoder == false
        DataLoader{T, typeof(data), Nothing, :TimeSeries}(data, nothing, dim2 * 2, time_steps, n_params, nothing, nothing)
    elseif autoencoder == true
        DataLoader{T, typeof(data), Nothing, :RegularData}(data, nothing, dim2 * 2, time_steps, n_params, nothing, nothing)
    end
end

function DataLoader(data::NamedTuple{(:q, :p), Tuple{VT, VT}}; 
                    suppress_info = false) where {VT <: AbstractVector}

    DataLoader((q = reshape(data.q, 1, length(data.q)), p = reshape(data.p, 1, length(data.p)));
                suppress_info = suppress_info)
end


function data_tensors_from_geometric_solution(solution::GeometricSolution{T, <:Number, NT}) where {T <: Number, DT <: DataSeries{T}, NT<:NamedTuple{(:q, :p), Tuple{DT, DT}}}
    sys_dim, input_time_steps = length(solution.s.q[0]), length(solution.t)
    data = (q = zeros(T, sys_dim, input_time_steps, 1), p = zeros(T, sys_dim, input_time_steps, 1))

    for dim in 1:sys_dim 
        data.q[dim, :, 1] = solution.q[:, dim]
        data.p[dim, :, 1] = solution.p[:, dim]
    end

    data
end

"""
    DataLoader(solution)

Make an instance of `DataLoader` for a `GeometricSolution`.

`GeometricSolution`s are imported from the the package [`GeometricSolutions.jl`](https://github.com/JuliaGNI/GeometricSolutions.jl).

# Arguments

This functor for [`DataLoader`](@ref) also has the keyword arguments 
- `autoencoder = false` and
- `suppress_info = false`. 
See the docstring for [`DataLoader(::AbstractArray{<:Number, 3})`](@ref).

# Implementation 

Internally this stores the data as a tensor where the third axis has length 1.
"""
function DataLoader(solution::GeometricSolution{T, <:Number, NT}; kwargs...) where {T <: Number, DT <: DataSeries{T}, NT<:NamedTuple{(:q, :p), Tuple{DT, DT}}}
    data = data_tensors_from_geometric_solution(solution)

    DataLoader(data; kwargs...)
end

function DataLoader(ensemble_solution::EnsembleSolution{T, T1, Vector{ST}};
                    suppress_info = false) where {T, 
                                    T1, 
                                    DT, 
                                    ST <: GeometricSolution{T, T1, NamedTuple{(:q, ), Tuple{DT}}}}

    sys_dim = length(ensemble_solution.s[1].q[0])
    input_time_steps = length(ensemble_solution.t)
    n_params = length(ensemble_solution.s)

    data = zeros(sys_dim, input_time_steps, n_params)

    for (solution, i) in zip(ensemble_solution.s, axes(ensemble_solution.s, 1))
        for dim in 1:sys_dim 
            data[dim, :, i] = solution.q[:, dim]
        end 
    end

    DataLoader(data; suppress_info = suppress_info)
end

"""
    DataLoader(ensemble_solution)

Make an instance of `DataLoader` for a `EnsembleSolution`.

`EnsembleSolution`s are imported from the the package [`GeometricSolutions.jl`](https://github.com/JuliaGNI/GeometricSolutions.jl).

# Arguments

This functor for `DataLoader` also has the keyword arguments 
- `autoencoder = false` and
- `suppress_info = false`. 

See the docstring for [`DataLoader(::AbstractArray{<:Number, 3})`](@ref).

# Implementation

Internally this stores the data as a tensor where the third axis has length equal to the number of solutions in the ensemble.
"""
function DataLoader(ensemble_solution::EnsembleSolution{T, T1, Vector{ST}}; 
                    autoencoder = false,
                    suppress_info = false
                    ) where {T, 
                             T1, 
                             DT <: DataSeries{T}, 
                             ST <: GeometricSolution{T, T1, NamedTuple{(:q, :p), Tuple{DT, DT}}}
                             }

    sys_dim = length(ensemble_solution.s[1].q[0])
    input_time_steps = length(ensemble_solution.t)
    n_params = length(ensemble_solution.s)

    data = (q = zeros(T, sys_dim, input_time_steps, n_params), p = zeros(T, sys_dim, input_time_steps, n_params))

    for (solution, i) in zip(ensemble_solution.s, axes(ensemble_solution.s, 1))
        for dim in 1:sys_dim 
            data.q[dim, :, i] = solution.q[:, dim]
            data.p[dim, :, i] = solution.p[:, dim]
        end 
    end

    DataLoader(data; autoencoder = autoencoder, suppress_info = suppress_info)
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

@doc raw"""
    DataLoader(dl, backend)

Make a new instance of `DataLoader` based on an existing instance of `DataLoader` for a new `backend`.

This allocates new memory of the same size as is used for the original `dl` and then copies the data.

By default the data type remains unchanged, i.e. `eltype(DataLoader(dl, backend)) == eltype(dl)` is `true`.

If you want to change the data type write e.g. 

```julia
    DataLoader(dl, backend, Float32)
```

# Arguments

There is an optional keyword argument 
- `autoencoder = nothing`. 

By default this inherits the autoencoder property form `dl`.

See the docstring for [`DataLoader(data::AbstractArray{<:Number, 3})`](@ref).
"""
function DataLoader(dl::DataLoader{T1, <:QPTOAT, Nothing, Type}, 
                    backend::KernelAbstractions.Backend=KernelAbstractions.get_backend(dl), 
                    T::DataType=T1; 
                    autoencoder = nothing                    
                    ) where {T1, Type}
    DT = if isnothing(autoencoder)
            Type
    elseif autoencoder == true
        :RegularData
    elseif autoencoder == false
        :TimeSeries
    end

    input = 
        if T == T1
            dl.input
        else
            map_to_type(dl.input, T)
        end

    new_input = 
        if backend == KernelAbstractions.get_backend(dl)
            input 
        else
            map_to_new_backend(input, backend)
        end

    DataLoader{T, typeof(new_input), Nothing, DT}(
        new_input,
        nothing,
        dl.input_dim,
        dl.input_time_steps,
        dl.n_params,
        nothing,
        nothing)
end

@doc raw"""
    accuracy(model, ps, dl)

Compute the accuracy of a neural network classifier.

This needs an instance of [`DataLoader`](@ref) that stores the *test data*.
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

@doc raw"""
    accuracy(nn, dl)

Compute the accuracy of a neural network classifier.

This is like [`accuracy(::Chain, ::Tuple, ::DataLoader)`](@ref), but for a `NeuralNetwork`.
"""
accuracy(nn::NeuralNetwork, dl::DataLoader) = accuracy(nn.model, nn.params, dl)

Base.eltype(::DataLoader{T}) where T = T

KernelAbstractions.get_backend(dl::DataLoader) = KernelAbstractions.get_backend(dl.input)
function KernelAbstractions.get_backend(dl::DataLoader{T, <:QPT{T}}) where T
    KernelAbstractions.get_backend(dl.input.q)
end