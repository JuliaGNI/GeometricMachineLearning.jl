description(::Val{:Batch}) = raw"""
`Batch` is a struct whose functor acts on an instance of `DataLoader` to produce a sequence of training samples for training for one epoch. 

## The Constructor

The constructor for `Batch` is called with: 
- `batch_size::Int`
- `seq_length::Int` (optional)
- `prediction_window::Int` (optional)

The first one of these arguments is required; it indicates the number of training samples in a batch. If we deal with time series data then we can additionaly supply a *sequence length* and a *prediction window* as input arguments to `Batch`. These indicate the number of input vectors and the number of output vectors.

## The functor 

An instance of `Batch` can be called on an instance of `DataLoader` to produce a sequence of samples that contain all the input data, i.e. for training for one epoch. The output of applying `batch:Batch` to `dl::DataLoader` is a tuple of vectors of integers. Each of these vectors contains two integers: the first is the *time index* and the second one is the *parameter index*.
"""

"""
$(description(Val(:Batch)))
"""
struct Batch{BatchType}
    batch_size::Int
    seq_length::Int
    prediction_window::Int
end

# add an additional constructor for `TransformerLoss` taking batch as input.
TransformerLoss(batch::Batch{:Transformer}) = TransformerLoss(batch.seq_length, batch.prediction_window)

Base.iterate(nn::NeuralNetwork, ics, batch::Batch{:Transformer}; n_points = 100) = iterate(nn, ics; n_points = n_points, prediction_window = batch.prediction_window)

function Batch(batch_size::Int)
    Batch{:FeedForward}(batch_size, 1, 1)
end

function Batch(batch_size::Int, seq_length::Int)
    Batch{:Transformer}(batch_size, seq_length, seq_length)
end

function Batch(batch_size::Int, seq_length::Int, prediction_window::Int)
    Batch{:Transformer}(batch_size, seq_length, prediction_window)
end

Batch(::Int, ::Nothing, ::Int) = error("Cannot provide prediction window alone. Need sequence length!")

@doc raw"""
Gives the number of batches. Inputs are of type `DataLoader` and `Batch`.

Here the big distinction is between data that are *time-series like* and data that are *autoencoder like*.
"""
function number_of_batches(dl::DataLoader{T, AT, OT, :TimeSeries}, batch::Batch) where {T, BT<:AbstractArray{T, 3}, AT<:Union{BT, NamedTuple{(:q, :p), Tuple{BT, BT}}}, OT}
    @assert dl.input_time_steps > (batch.seq_length + batch.prediction_window) "The number of time steps has to be greater than sequence length + prediction window."
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

    batch_indices = KernelAbstractions.allocate(backend, Int, 2, _batch_size)
    batch_indices_temp = zeros(Int, size(batch_indices)...)
    for t in axes(batch_indices_tuple, 1)
        batch_indices_temp[1, t] = batch_indices_tuple[t][1]
        batch_indices_temp[2, t] = batch_indices_tuple[t][2]
    end
    batch_indices = typeof(batch_indices)(batch_indices_temp)

    batch_indices
end

"""
Takes the output of the batch functor and uses it to create the corresponding array (NamedTuples). 
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

"""
Takes the output of the batch functor and uses it to create the corresponding array. 
"""
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
function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT, OT}, ::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, T1, BT<:AbstractArray{T, 3}, OT<:AbstractArray{T1, 3}}
    _batch_indices = [batch_index[1] for batch_index in batch_indices_tuple]
    input_batch = copy(dl.input[:, :, _batch_indices])
    output_batch = copy(dl.output[:, :, _batch_indices])

    input_batch, output_batch
end