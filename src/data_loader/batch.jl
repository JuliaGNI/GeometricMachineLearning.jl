description(::Val{:Batch}) = raw"""
`Batch` is a struct whose functor acts on an instance of `DataLoader` to produce a sequence of training samples for one epoch. 

## The Constructor

The constructor for `Batch` is called with: 
- `batch_size::Int`
- `seq_length::Int` (optional)
- `prediction_window::Int` (optional)

## The functor 

An instance of `Batch` can be called on an instance of `DataLoader` to produce a sequence of samples that contain all the input data, i.e. for training for one epoch. 
"""

description(::Val{:batch_functor_matrix}) = raw"""
For a snapshot matrix (or a `NamedTuple` of the form `(q=A, p=B)` where `A` and `B` are matrices), the functor for `Batch` is called on an instance of `DataLoader`. It then returns a tuple of batch indices: 
- for `autoencoder=true`: ``(\mathcal{I}_1, \ldots, \mathcal{I}_{\lceil\mathtt{n\_params/batch\_size}\rceil})``, where the index runs from 1 to the number of batches, which is the number of columns in the snapshot matrix divided by the batch size (rounded up).
- for `autoencoder=false`: ``(\mathcal{I}_1, \ldots, \mathcal{I}_{\lceil\mathtt{dl.input\_time\_steps/batch\_size}\rceil})``, where the index runs from 1 to the number of batches, which is the number of columns in the snapshot matrix (minus one) divided by the batch size (rounded up).
"""

description(::Val{:batch_functor_tensor}) = raw"""
The functor for batch is called with an instance on `DataLoader`. It then returns a tuple of batch indices: ``(\mathcal{I}_1, \ldots, \mathcal{I}_{\lceil\mathtt{dl.n\_params/batch\_size}\rceil})``, where the index runs from 1 to the number of batches, which is the number of parameters divided by the batch size (rounded up).
"""

"""
## `Batch` constructor
$(description(Val(:Batch)))

## `Batch functor`
$(description(Val(:batch_functor_matrix)))

$(description(Val(:batch_functor_tensor)))
"""
struct Batch{seq_type <: Union{Nothing, Int}}
    batch_size::Int
    seq_length::seq_type
    prediction_window::seq_type
end

# add an additional constructor for `TransformerLoss` taking batch as input.
TransformerLoss(batch::Batch{Int}) = TransformerLoss(batch.seq_length, batch.prediction_window)

Base.iterate(nn::NeuralNetwork, ics, batch::Batch{Int}; n_points = 100) = iterate(nn, ics; n_points = n_points, prediction_window = batch.prediction_window)

function Batch(batch_size, seq_length = nothing, prediction_window = nothing)
    Batch{typeof(seq_length)}(batch_size, seq_length, prediction_window)
end

# if no prediction window is provided it is set to seq_length.
function Batch(batch_size::Int, seq_length::Int, ::Nothing)
    Batch{typeof(seq_length)}(batch_size, seq_length, seq_length)
end

Batch(::Int, ::Nothing, ::Int) = error("Cannot provide prediction window alone. Need sequence length!")

hasseqlength(::Batch{<:Integer}) = true
hasseqlength(::Batch{<:Nothing}) = false

@doc raw"""
This function is called when either dealing with a matrix or a tensor which **is not** of time series nature. It takes an input a 
- `batch::Batch`
- `number_columns::Int`

The output is a *tuple of vectors of indices*: 
``([i^1_1, i^1_2, \ldots, i^1_\mathrm{bs}], \ldots, [i^{\ell - 1}_1, i^{\ell - 1}_2, \ldots, i^{\ell - 1}_\mathrm{bs}], [i^\ell_1, i^\ell_2, \ldots, i^\ell_\mathrm{r}])`` where ``r`` is the remainder.  
"""
function batch_over_one_axis(batch::Batch, number_columns::Int)
    indices = shuffle(1:number_columns)
    n_batches = Int(ceil(number_columns / batch.batch_size))
    batches = ()
    for batch_number in 1:(n_batches - 1)
        batches = (batches..., indices[(batch_number - 1) * batch.batch_size + 1 : batch_number * batch.batch_size])
    end
    # this last line is needed if the number of columns is not divisible by the batch size. 
    (batches..., indices[(n_batches - 1) * batch.batch_size + 1:number_columns])
end

function batch_over_two_axis(batch::Batch, number_columns::Int, third_dim::Int)
    time_indices = shuffle(1:number_columns)
    parameter_indices = shuffle(1:third_dim)
    complete_indices = Iterators.product(time_indices, parameter_indices) |> collect |> vec
    batches = ()
    n_batches = Int(ceil(number_columns * third_dim / batch.batch_size))
    for batch_number in 1:(n_batches - 1)
        batches = (batches..., complete_indices[(batch_number - 1) * batch.batch_size + 1 : batch_number * batch.batch_size])
    end
    (batches..., complete_indices[(n_batches - 1) * batch.batch_size + 1:end])
end

# (::Batch{Nothing})(::DataLoader{T, AT, Nothing, TimeSteps}) where {T <: Number, AT} = error("Need to provide `seq_length` when dealing with time series data.")

function (batch::Batch{<:Nothing})(dl::DataLoader{T, BT, OT, RegularData}) where {T, AT<:AbstractArray{T}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}, OT}
    batch_over_one_axis(batch, dl.n_params)
end

function (batch::Batch{<:Nothing})(dl::DataLoader{T, BT, OT, TimeSteps}) where {T, AT<:AbstractArray{T}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}, OT}
    batch_over_one_axis(batch, dl.input_time_steps)
end

function (batch::Batch{Int})(dl::DataLoader{T, BT, OT, TimeSteps}) where {T, AT<:AbstractArray{T}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}, OT}
    batch_over_one_axis(batch, dl.input_time_steps - batch.seq_length - batch.prediction_window)
end

function (batch::Batch{Nothing})(dl::DataLoader{T, BT, Nothing}) where {T, AT<:AbstractArray{T, 3}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}}
    batch_over_two_axis(batch, dl.input_time_steps, dl.n_params)
end 

# Batching for tensor with three axes (unsupervised learning). 
function (batch::Batch{Int})(dl::DataLoader{T, BT, Nothing}) where {T, AT<:AbstractArray{T, 3}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}}
    batch_over_two_axis(batch, dl.input_time_steps - batch.seq_length - batch.prediction_window, dl.n_params)
end 

@doc raw"""
Gives the number of batches. Inputs are of type `DataLoader` and `Batch`.
"""
function number_of_batches(dl::DataLoader{T, AT}, batch::Batch{Nothing}) where {T, BT<:AbstractMatrix{T}, AT<:Union{BT, NamedTuple{(:q, :p), Tuple{BT, BT}}}}
    Int(ceil(dl.input_time_steps / batch.batch_size))
end

# function number_of_batches(dl::DataLoader{T, AT}, batch::Batch{Nothing}) where {T, BT<:AbstractArray{T, 3}, AT<:Union{BT, NamedTuple{(:q, :p), Tuple{BT, BT}}}}
#     Int(ceil(dl.n_params / batch.batch_size))
# end

function number_of_batches(dl::DataLoader{T, AT}, batch::Batch{Int}) where {T, BT<:AbstractArray{T, 3}, AT<:Union{BT, NamedTuple{(:q, :p), Tuple{BT, BT}}}}
    Int(ceil((dl.input_time_steps - batch.seq_length - batch.prediction_window) * dl.n_params / batch.batch_size))
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

"""
Takes the output of the batch functor and uses it to create the corresponding array (NamedTuples). 
"""
function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT}, batch::Batch{Int}, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, AT<:AbstractArray{T, 3}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}
    backend = KernelAbstractions.get_backend(dl.input.q)
    q_input = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, batch.seq_length, length(batch_indices_tuple))
    p_input = similar(q_input)

    batch_indices = KernelAbstractions.allocate(backend, Int, 2, length(batch_indices_tuple))
    batch_indices_temp = zeros(Int, size(batch_indices)...)
    for t in axes(batch_indices_tuple, 1)
        batch_indices_temp[1, t] = batch_indices_tuple[t][1]
        batch_indices_temp[2, t] = batch_indices_tuple[t][2]
    end
    batch_indices = typeof(batch_indices)(batch_indices_temp)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(q_input, p_input, dl.input, batch_indices, ndrange=(dl.input_dim ÷ 2, batch.seq_length, batch.prediction_window, length(batch_indices_tuple)))

    q_output = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, batch.prediction_window, length(batch_indices_tuple))
    p_output = similar(q_output)

    assign_output_from_vector_of_tuples! = assign_output_from_vector_of_tuples_kernel!(backend)
    assign_output_from_vector_of_tuples!(q_output, p_output, dl.input, batch_indices, batch.seq_length, ndrange=(dl.input_dim ÷ 2, batch.prediction_window, length(batch_indices_tuple)))

    (q = q_input, p = p_input), (q = q_output, p = p_output)
end

"""
Takes the output of the batch functor and uses it to create the corresponding array. 
"""
function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, BT<:AbstractArray{T, 3}}
    backend = KernelAbstractions.get_backend(dl.input)
    input = KernelAbstractions.allocate(backend, T, dl.input_dim, batch.seq_length, length(batch_indices_tuple))

    batch_indices = KernelAbstractions.allocate(backend, Int, 2, length(batch_indices_tuple))
    batch_indices_temp = zeros(Int, size(batch_indices)...)
    for t in axes(batch_indices_tuple, 1)
        batch_indices_temp[1, t] = batch_indices_tuple[t][1]
        batch_indices_temp[2, t] = batch_indices_tuple[t][2]
    end
    batch_indices = typeof(batch_indices)(batch_indices_temp)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(input, dl.input, batch_indices, ndrange=(dl.input_dim, batch.seq_length, length(batch_indices_tuple)))

    output = KernelAbstractions.allocate(backend, T, dl.input_dim, batch.prediction_window, length(batch_indices_tuple))

    assign_output_from_vector_of_tuples! = assign_output_from_vector_of_tuples_kernel!(backend)
    assign_output_from_vector_of_tuples!(output, dl.input, batch_indices, batch.seq_length, ndrange=(dl.input_dim, batch.prediction_window, length(batch_indices_tuple)))

    input, output
end