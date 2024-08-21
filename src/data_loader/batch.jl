@doc raw"""
`Batch` is a struct whose functor acts on an instance of `DataLoader` to produce a sequence of training samples for training for one epoch. 

See [`Batch(::Int)`](@ref), [`Batch(::Int, ::Int)`](@ref) and [`Batch(::Int, ::Int, ::Int)`](@ref) for the different constructors.

# The functor 

An instance of `Batch` can be called on an instance of `DataLoader` to produce a sequence of samples that contain all the input data, i.e. for training for one epoch. 

The output of applying `batch:Batch` to `dl::DataLoader` is a tuple of vectors of integers. Each of these vectors contains two integers: the first is the *time index* and the second one is the *parameter index*.

# Examples

Consider the following example for drawing batches of size 2 for an instance of `DataLoader` constructed with a vector:

```jldoctest
using GeometricMachineLearning
import Random

Random.seed!(123)

dl = DataLoader(rand(5))
batch = Batch(2)

batch(dl)

# output

[ Info: You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters.
([(1, 4), (1, 3)], [(1, 2), (1, 1)], [(1, 5)])
```

Here the first index is always 1 (the time dimension). We get a total number of 3 batches. 
The last batch is only of size 1 because we *sample without replacement*.
Also see the docstring for [`DataLoader(::AbstractVector)`](@ref).
"""
struct Batch{BatchType}
    batch_size::Int
    seq_length::Int
    prediction_window::Int
end

# add an additional constructor for `TransformerLoss` taking batch as input.
TransformerLoss(batch::Batch{:Transformer}) = TransformerLoss(batch.seq_length, batch.prediction_window)

Base.iterate(nn::NeuralNetwork, ics, batch::Batch{:Transformer}; n_points = 100) = iterate(nn, ics; n_points = n_points, prediction_window = batch.prediction_window)

@doc raw"""
    Batch(batch_size)

Make an instance of `Batch` for a specific batch size.

This is used to train neural networks of `FeedForward` type (as opposed to transformers).
"""
function Batch(batch_size::Int)
    Batch{:FeedForward}(batch_size, 1, 1)
end

@doc raw"""
    Batch(batch_size, seq_length)

Make an instance of `Batch` for a specific batch size and a sequence length.

This is used to train neural networks of `Transformer` type.

Optionally the prediction window can also be specified by calling:

```jldoctest
using GeometricMachineLearning

batch_size = 2
seq_length = 3
prediction_window = 2

Batch(batch_size, seq_length, prediction_window)

# output

Batch{:Transformer}(2, 3, 2)
```

Note that here the batch is of type `:Transformer`.
"""
function Batch(batch_size::Int, seq_length::Int, prediction_window::Int=seq_length)
    Batch{:Transformer}(batch_size, seq_length, prediction_window)
end

Batch(::Int, ::Nothing, ::Int) = error("Cannot provide prediction window alone. Need sequence length!")

function Batch(batch_size::Int, dl::DataLoader{T, AT, OT, :TimeSeries}) where {T, T1, AT<:AbstractArray{T, 3}, OT<:AbstractArray{T1, 3}}
    Batch(batch_size, dl.input_time_steps, 0)
end

@doc raw"""
    number_of_batches(dl, batch)

Compute the number of batches.

Here the big distinction is between data that are *time-series like* and data that are *autoencoder like*.
"""
function number_of_batches(dl::DataLoader{T, AT, OT, :TimeSeries}, batch::Batch) where {T, BT<:AbstractArray{T, 3}, AT<:Union{BT, NamedTuple{(:q, :p), Tuple{BT, BT}}}, OT}
    @assert dl.input_time_steps ≥ (batch.seq_length + batch.prediction_window) "The number of time steps has to be greater than sequence length + prediction window."
    Int(ceil((dl.input_time_steps - (batch.seq_length - 1) - batch.prediction_window) * dl.n_params / batch.batch_size))
end

function number_of_batches(dl::DataLoader{T, AT, OT, :RegularData}, batch::Batch) where {T, BT<:AbstractArray{T, 3}, AT<:Union{BT, NamedTuple{(:q, :p), Tuple{BT, BT}}}, OT}
    Int(ceil(dl.input_time_steps * dl.n_params / batch.batch_size))
end

function batch_over_two_axes(batch::Batch, number_columns::Int, third_dim::Int, dl::DataLoader)
    time_indices = shuffle(1:number_columns)
    parameter_indices = shuffle(1:third_dim)
    complete_indices = Iterators.product(time_indices, parameter_indices) |> collect |> vec
    batches = ()
    n_batches = number_of_batches(dl, batch)
    for batch_number in 1:(n_batches - 1)
        batches = (batches..., complete_indices[(batch_number - 1) * batch.batch_size + 1 : batch_number * batch.batch_size])
    end
    (batches..., complete_indices[(n_batches - 1) * batch.batch_size + 1:end])
end

function (batch::Batch)(dl::DataLoader{T, BT, OT, :RegularData}) where {T, AT<:AbstractArray{T, 3}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}, OT}
    batch_over_two_axes(batch, dl.input_time_steps, dl.n_params, dl)
end

function (batch::Batch)(dl::DataLoader{T, BT, OT, :TimeSeries}) where {T, AT<:AbstractArray{T, 3}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}, OT}
    batch_over_two_axes(batch, dl.input_time_steps - (batch.seq_length - 1) - batch.prediction_window, dl.n_params, dl)
end

@kernel function assign_input_from_vector_of_tuples_kernel!(q_input::AT, p_input::AT, input::NamedTuple{(:q, :p), Tuple{AT, AT}}, indices::AbstractArray{Int, 2}) where {T, AT<:AbstractArray{T, 3}}
    i, j, k = @index(Global, NTuple)
    
    q_input[i, j, k] = input.q[i, indices[1, k] + j - 1, indices[2, k]]
    p_input[i, j, k] = input.p[i, indices[1, k] + j - 1, indices[2, k]]
end

@kernel function assign_input_from_vector_of_tuples_kernel!(input::AT, data::AT, indices::AbstractArray{Int, 2}) where {T, AT<:AbstractArray{T, 3}}
    i, j, k = @index(Global, NTuple)
    
    input[i, j, k] = data[i, indices[1, k] + j - 1, indices[2, k]]
end

@kernel function assign_output_from_vector_of_tuples_kernel!(q_output::AT, p_output::AT, input::NamedTuple{(:q, :p), Tuple{AT, AT}}, indices::AbstractArray{Int, 2}, seq_length::Int) where {T, AT<:AbstractArray{T, 3}}
    i, j, k = @index(Global, NTuple)

    q_output[i, j, k] = input.q[i, indices[1, k] + seq_length + j - 1, indices[2, k]]
    p_output[i, j, k] = input.p[i, indices[1, k] + seq_length + j - 1, indices[2, k]]
end

@kernel function assign_output_from_vector_of_tuples_kernel!(output::AT, data::AT, indices::AbstractArray{Int, 2}, seq_length::Int) where {T, AT<:AbstractArray{T, 3}}
    i, j, k = @index(Global, NTuple)

    output[i, j, k] = data[i, indices[1, k] + seq_length + j - 1, indices[2, k]]
end

# this is neeced if we want to use the vector of tuples in a kernel
function convert_vector_of_tuples_to_matrix(backend::Backend, batch_indices_tuple::Vector{Tuple{Int, Int}})
    _batch_size = length(batch_indices_tuple)

    batch_indices = KernelAbstractions.zeros(backend, Int, 2, _batch_size)
    @views for t in axes(batch_indices_tuple, 1)
        batch_indices[1, t] = batch_indices_tuple[t][1]
        batch_indices[2, t] = batch_indices_tuple[t][2]
    end

    batch_indices
end

@doc raw"""
    convert_input_and_batch_indices_to_array(dl, batch, batch_indices)

Assign batch data based on batch indices.

# Examples

```jldoctest
using GeometricMachineLearning

dl = DataLoader([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
batch = Batch(3)
batch_indices = [(1, 1), (1, 3), (1, 5)]

GeometricMachineLearning.convert_input_and_batch_indices_to_array(dl, batch, batch_indices)

# output

[ Info: You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters.
1×1×3 Array{Float64, 3}:
[:, :, 1] =
 0.1

[:, :, 2] =
 0.3

[:, :, 3] =
 0.5
```
"""
function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, AT<:AbstractArray{T, 3}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}
    backend = KernelAbstractions.get_backend(dl.input.q)
    
    # the batch size is smaller for the last batch
    _batch_size = length(batch_indices_tuple)

    batch_indices = convert_vector_of_tuples_to_matrix(backend, batch_indices_tuple)

    q_input = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, batch.seq_length, _batch_size)
    p_input = similar(q_input)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(q_input, p_input, dl.input, batch_indices, ndrange=(dl.input_dim ÷ 2, batch.seq_length, _batch_size))

    q_output = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, batch.prediction_window, _batch_size)
    p_output = similar(q_output)

    assign_output_from_vector_of_tuples! = assign_output_from_vector_of_tuples_kernel!(backend)
    assign_output_from_vector_of_tuples!(q_output, p_output, dl.input, batch_indices, batch.seq_length, ndrange=(dl.input_dim ÷ 2, batch.prediction_window, _batch_size))

    (q = q_input, p = p_input), (q = q_output, p = p_output)
end

function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, BT<:AbstractArray{T, 3}}
    backend = KernelAbstractions.get_backend(dl.input)

    # the batch size is smaller for the last batch 
    _batch_size = length(batch_indices_tuple)

    batch_indices = convert_vector_of_tuples_to_matrix(backend, batch_indices_tuple)

    input = KernelAbstractions.allocate(backend, T, dl.input_dim, batch.seq_length, _batch_size)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(input, dl.input, batch_indices, ndrange=(dl.input_dim, batch.seq_length, _batch_size))

    output = KernelAbstractions.allocate(backend, T, dl.input_dim, batch.prediction_window, _batch_size)

    assign_output_from_vector_of_tuples! = assign_output_from_vector_of_tuples_kernel!(backend)
    assign_output_from_vector_of_tuples!(output, dl.input, batch_indices, batch.seq_length, ndrange=(dl.input_dim, batch.prediction_window, _batch_size))

    input, output
end

function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT, Nothing, :RegularData}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, AT<:AbstractArray{T, 3}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}
    backend = KernelAbstractions.get_backend(dl.input.q)

    # the batch size is smaller for the last batch
    _batch_size = length(batch_indices_tuple)

    batch_indices = convert_vector_of_tuples_to_matrix(backend, batch_indices_tuple)

    q_input = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, batch.seq_length, _batch_size)
    p_input = similar(q_input)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(q_input, p_input, dl.input, batch_indices, ndrange=(dl.input_dim ÷ 2, batch.seq_length, _batch_size))

    (q = q_input, p = p_input)
end

function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT, Nothing, :RegularData}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, BT<:AbstractArray{T, 3}}
    backend = KernelAbstractions.get_backend(dl.input)

    # the batch size is smaller for the last batch 
    _batch_size = length(batch_indices_tuple)

    batch_indices = convert_vector_of_tuples_to_matrix(backend, batch_indices_tuple)

    input = KernelAbstractions.allocate(backend, T, dl.input_dim, batch.seq_length, _batch_size)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(input, dl.input, batch_indices, ndrange=(dl.input_dim, batch.seq_length, _batch_size))

    input
end

# for the case when the DataLoader also contains an output
function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT, OT}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, T1, BT<:AbstractArray{T, 3}, OT<:AbstractArray{T1, 3}}
    time_indices = [batch_index[1] for batch_index in batch_indices_tuple]
    parameter_indices = [batch_index[2] for batch_index in batch_indices_tuple]
    @views input_batch = dl.input[:, :, parameter_indices]
    @views output_batch = dl.output[:, :, parameter_indices]

    input_batch, output_batch
end

function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT, BT}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, BT<:AbstractArray{T, 3}}
    backend = KernelAbstractions.get_backend(dl.input)

    # the batch size is smaller for the last batch 
    _batch_size = length(batch_indices_tuple)

    batch_indices = convert_vector_of_tuples_to_matrix(backend, batch_indices_tuple)

    input = KernelAbstractions.allocate(backend, T, dl.input_dim, batch.seq_length, _batch_size)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(input, dl.input, batch_indices, ndrange=(dl.input_dim, batch.seq_length, _batch_size))

    output = KernelAbstractions.allocate(backend, T, dl.output_dim, batch.prediction_window, _batch_size)

    assign_input_from_vector_of_tuples!(output, dl.output, batch_indices, ndrange=(dl.output_dim, batch.prediction_window, _batch_size))

    input, output
end