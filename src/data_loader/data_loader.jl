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

[ Info: You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of time steps.
2×3×1 Array{Int64, 3}:
[:, :, 1] =
 1  2  3
 4  5  6
```
"""
struct DataLoader{T, AT <: Union{NamedTuple, AbstractArray{T}},
    OT <: Union{AbstractArray, Nothing}, K, S <: Union{Int, Nothing}}
    input::AT
    output::OT
    input_dim::Int
    input_time_steps::Int
    n_params::Int
    output_dim::S
    output_time_steps::S
end

# The fifth parameter is the type of the two output fields, which is `Nothing` for a loader without
# an output and `Int` with one. A caller that names the first four — `DataLoader{T, AT, OT,
# :TimeSeries}(…)`, which is how `GMLDatasets` builds a loader — gets it filled in from the fields,
# and the five counts are converted to `Int` whatever integer type the caller passes.
function DataLoader{T, AT, OT, K}(input::AT, output::OT, input_dim, input_time_steps,
        n_params, output_dim, output_time_steps) where {T, AT, OT, K}
    S = output_dim === nothing ? Nothing : Int
    DataLoader{T, AT, OT, K, S}(input, output, input_dim, input_time_steps, n_params,
        output_dim, output_time_steps)
end

# The two decisions `autoencoder` makes at every constructor that takes a data set: the kind it
# builds, and the two lengths of a matrix or a `(q, p)` pair read as a set of samples, whose second
# axis is then the parameter index and whose time axis has length one.
data_set_kind(autoencoder::Bool) = autoencoder ? :RegularData : :TimeSeries
data_set_shape(autoencoder::Bool, n::Int) = autoencoder ? (1, n) : (n, 1)

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
DataLoader{Int64, Array{Int64, 3}, Nothing, :TimeSeries, Nothing}([1 4; 2 5; 3 6;;; 7 10; 8 11; 9 12], nothing, 3, 2, 2, nothing, nothing)
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

DataLoader{Int64, Array{Int64, 3}, Nothing, :TimeSeries, Nothing}([1 4; 2 5; 3 6;;; 7 10; 8 11; 9 12], nothing, 3, 2, 2, nothing, nothing)
```
the `@info` statement is not printed.
"""
Base.@constprop :aggressive function DataLoader(
        data::AbstractArray{T, 3}; autoencoder::Bool = false, suppress_info = false) where {T}
    if !suppress_info
        @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of time steps and (iii) number of params."
    end
    input_dim, input_time_steps, n_params = size(data)

    DataLoader{T, typeof(data), Nothing, data_set_kind(autoencoder)}(
        data, nothing, input_dim, input_time_steps, n_params, nothing, nothing)
end

"""
    DataLoader(data::AbstractMatrix)

Make an instance of [`DataLoader`](@ref) based on a matrix.

# Arguments

See [`DataLoader(::AbstractArray{<:Number, 3})`](@ref) for details.

# Implementation

The second axis of the matrix is the number of time steps, and internally the data are reshaped to
a tensor of shape `(size(data)..., 1)`: one trajectory of one parameter. With `autoencoder = true`
the matrix is a set of samples instead, so that axis is the parameter index and the time axis has
length one.
"""
function DataLoader(
        data::AbstractMatrix{T}; autoencoder::Bool = false, suppress_info = false) where {T}
    if !suppress_info
        @info "You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) $(autoencoder ? "number of parameters" : "number of time steps")."
    end
    input_dim, second_dim = size(data)
    time_steps, n_params = data_set_shape(autoencoder, second_dim)
    DataLoader(reshape(data, input_dim, time_steps, n_params);
        autoencoder = autoencoder, suppress_info = true)
end

"""
    DataLoader(data::AbstractVector)

Make an instance of `DataLoader` based on a vector.

# Extended help

If the input to `DataLoader` is a vector, it is assumed that this vector represents one-dimensional time-series data and is therefore processed as:

```julia
    DataLoader(data::AbstractVector; autoencoder=false) = DataLoader(reshape(data, 1, length(data)); autoencoder = autoencoder)
```
"""
function DataLoader(data::AbstractVector; autoencoder::Bool = false, suppress_info = false)
    DataLoader(reshape(data, 1, length(data)); autoencoder = autoencoder, suppress_info = suppress_info)
end

# A loader that holds an output is a set of samples whatever its kind: its target is the output
# itself and never the input shifted by the sequence length. The output's element type is the
# input's or an integer, so a real input can carry an integer one-hot target — a classifier's data,
# and `accuracy`'s argument — and any other pair of element types is a `MethodError`.
function DataLoader(input::AbstractArray{T, 3}, output::AbstractArray{T1, 3};
        suppress_info = false) where {T, T1 <: Union{T, Integer}}
    @assert size(input, 3) == size(output, 3)
    if !suppress_info
        @info "You have provided an input and an output."
    end

    DataLoader{T, typeof(input), typeof(output), :RegularData}(
        input, output, size(input, 1), size(input, 2),
        size(input, 3), size(output, 1), size(output, 2))
end

function DataLoader(input::AbstractMatrix{T}, output::AbstractMatrix{T}; suppress_info = false) where {T}
    DataLoader(reshape(input, size(input)..., 1),
        reshape(output, size(output)..., 1); suppress_info = suppress_info)
end

function DataLoader(input::AbstractVector{T}, output::AbstractVector{T}; suppress_info = false) where {T}
    DataLoader(reshape(input, 1, length(input)),
        reshape(output, 1, length(output)); suppress_info = suppress_info)
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
        autoencoder::Bool = false,
        suppress_info = false) where {T, AT <: AbstractMatrix{T}}
    if !suppress_info
        @info "You have provided a NamedTuple with keys q and p; the data are matrices. This is interpreted as *symplectic data*."
    end
    dim2, second_dim = size(data.q)
    time_steps, n_params = data_set_shape(autoencoder, second_dim)
    DataLoader(map(tensor -> reshape(tensor, dim2, time_steps, n_params), data);
        autoencoder = autoencoder, suppress_info = true)
end

Base.@constprop :aggressive function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}};
        autoencoder::Bool = false,
        suppress_info = false) where {T, AT <: AbstractArray{T, 3}}
    if !suppress_info
        @info "You have provided a NamedTuple with keys q and p; the data are tensors. This is interpreted as *symplectic data*."
    end
    dim2, time_steps, n_params = size(data.q)

    DataLoader{T, typeof(data), Nothing, data_set_kind(autoencoder)}(
        data, nothing, dim2 * 2, time_steps, n_params, nothing, nothing)
end

function DataLoader(data::NamedTuple{(:q, :p), Tuple{VT, VT}};
        autoencoder::Bool = false, suppress_info = false) where {VT <: AbstractVector}
    DataLoader(
        (q = reshape(data.q, 1, length(data.q)), p = reshape(data.p, 1, length(data.p)));
        autoencoder = autoencoder, suppress_info = suppress_info)
end

# The 1-based tensor of one coordinate of one solution, or of each member of an ensemble along the
# parameter axis, with the axes `(system dimension, time steps, parameters)`.
function states_tensor(states::AbstractVector, n_params::Int = 1)
    Array{eltype(first(states))}(undef, length(first(states)), length(states), n_params)
end

# The states of one coordinate into the column `i` of `data`. The `q` of a solution carries the
# `DataSeries` time axis `0:n` and a `DataLoader` tensor is 1-based on all three axes, so the states
# are copied in one at a time and no whole coordinate is built on the way.
function data_tensor!(data, states::AbstractVector, i::Int)
    for (t, state) in enumerate(states)
        data[:, t, i] = state
    end
    data
end

data_tensor(states::AbstractVector) = data_tensor!(states_tensor(states), states, 1)

# The tensor of one coordinate over the members of an ensemble, along the parameter axis.
function ensemble_data_tensor(solutions, coordinate::Symbol)
    data = states_tensor(getproperty(first(solutions), coordinate), length(solutions))
    foreach(i -> data_tensor!(data, getproperty(solutions[i], coordinate), i),
        eachindex(solutions))
    data
end

function data_tensors_from_geometric_solution(
        solution::GeometricSolution{
        <:Number, <:Number, <:Any, <:NamedTuple{(:t, :q, :p, :q̇, :ṗ)}})
    (q = data_tensor(solution.q), p = data_tensor(solution.p))
end

function data_tensors_from_geometric_solution(solution::GeometricSolution{
        <:Number, <:Number,
        <:Any, <:Union{NamedTuple{(:t, :q, :v)}, NamedTuple{(:t, :q, :q̇)}}})
    data_tensor(solution.q)
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
function DataLoader(solution::GeometricSolution{T, T2, TT, NT};
        autoencoder::Bool = false, suppress_info = false) where {T <: Number, T2 <: Number,
        TT <: TimeSeries{T2}, TuT,
        NT <: Union{NamedTuple{(:t, :q, :p, :q̇, :ṗ), TuT},
            NamedTuple{(:t, :q, :v), TuT}, NamedTuple{(:t, :q, :q̇), TuT}}}
    DataLoader(data_tensors_from_geometric_solution(solution);
        autoencoder = autoencoder, suppress_info = suppress_info)
end

function DataLoader(ensemble_solution::EnsembleSolution{T, T1, Vector{ST}};
        autoencoder::Bool = false, suppress_info = false) where {T, T1, TuT,
        TT <: TimeSeries{T1}, ST <: Union{
            GeometricSolution{T, T1, TT, NamedTuple{(:t, :q, :v), TuT}},
            GeometricSolution{T, T1, TT, NamedTuple{(:t, :q, :q̇), TuT}}}}
    DataLoader(ensemble_data_tensor(ensemble_solution.s, :q);
        autoencoder = autoencoder, suppress_info = suppress_info)
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
        autoencoder::Bool = false, suppress_info = false) where {T, T1, TuT,
        TT <: TimeSeries{T1}, ST <: GeometricSolution{T, T1, TT,
            NamedTuple{(:t, :q, :p, :q̇, :ṗ), TuT}}}
    DataLoader(
        (q = ensemble_data_tensor(ensemble_solution.s, :q),
            p = ensemble_data_tensor(ensemble_solution.s, :p));
        autoencoder = autoencoder, suppress_info = suppress_info)
end

# `Base.copyto!` carries a host array to any backend; `KernelAbstractions.copyto!` has a method for
# the CPU alone, and a device backend such as `JLArray`'s has none.
function map_to_new_backend(input::AbstractArray{T},
        backend::KernelAbstractions.Backend) where {T}
    converted = KernelAbstractions.allocate(backend, T, size(input)...)
    Base.copyto!(converted, input)
    converted
end

# `map` over a `(q, p)` named tuple reaches each of the two tensors; over a plain array it would
# map the array's elements instead, which is why an array has a method of its own.
function map_to_new_backend(input::QPT, backend::KernelAbstractions.Backend)
    map(tensor -> map_to_new_backend(tensor, backend), input)
end

map_to_type(input::AbstractArray, ::Type{T}) where {T} = T.(input)

map_to_type(input::QPT, ::Type{T}) where {T} = map(tensor -> map_to_type(tensor, T), input)

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
function DataLoader(dl::DataLoader{T1, <:QPTOAT, Nothing, K},
        backend::KernelAbstractions.Backend = networkbackend(dl),
        ::Type{T} = T1;
        autoencoder::Union{Nothing, Bool} = nothing
) where {T1, K, T}
    kind = isnothing(autoencoder) ? K : data_set_kind(autoencoder)
    input = T == T1 ? dl.input : map_to_type(dl.input, T)
    input = backend == networkbackend(dl) ? input : map_to_new_backend(input, backend)

    DataLoader{T, typeof(input), Nothing, kind}(
        input, nothing, dl.input_dim, dl.input_time_steps, dl.n_params, nothing, nothing)
end

function DataLoader(dl::DataLoader, ::Type{T}; kwargs...) where {T}
    DataLoader(dl, networkbackend(dl), T; kwargs...)
end

@doc raw"""
    accuracy(model, ps, dl)

Compute the accuracy of a neural network classifier.

This needs an instance of [`DataLoader`](@ref) that stores the *test data*.
"""
function accuracy(model::Chain,
        ps::NetworkParameters,
        dl::DataLoader{T, AT, BT}) where {
        T, T1 <: Integer, AT <: AbstractArray{T}, BT <: AbstractArray{T1}}
    output_tensor = model(dl.input, ps)
    output_estimate = assign_output_estimate(output_tensor, dl.output_time_steps)
    backend = networkbackend(output_estimate)
    tensor_of_maximum_elements = KernelAbstractions.zeros(backend, T1, size(output_estimate)...)
    ind = argmax(output_estimate, dims = 1)
    # get tensor of maximum elements
    tensor_of_maximum_elements[ind] .= T1(1)
    (size(dl.output, 3) - sum(abs.(dl.output - tensor_of_maximum_elements)) / T1(2)) /
    size(dl.output, 3)
end

@doc raw"""
    accuracy(nn, dl)

Compute the accuracy of a neural network classifier.

This is like [`accuracy(::Chain, ::Tuple, ::DataLoader)`](@ref), but for a `NeuralNetwork`.
"""
accuracy(nn::NeuralNetwork, dl::DataLoader) = accuracy(nn.model, params(nn), dl)

Base.eltype(::DataLoader{T}) where {T} = T

networkbackend(dl::DataLoader) = networkbackend(dl.input)
function networkbackend(dl::DataLoader{T, <:QPT{T}}) where {T}
    networkbackend(dl.input.q)
end
