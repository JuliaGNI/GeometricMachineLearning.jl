@doc raw"""
    Batch

`Batch` is a struct whose functor acts on an instance of `DataLoader` to produce a sequence of training samples for training for one epoch. 

See [`Batch(::Int)`](@ref) and [`Batch(::Int, ::Int, ::Int)`](@ref) for the different constructors.

# The functor 

An instance of `Batch` can be called on an instance of `DataLoader` to produce a sequence of samples that contain all the input data, i.e. for training for one epoch. 

The output of applying `batch:Batch` to `dl::DataLoader` is a vector of vectors of integers. Each of these vectors contains two integers: the first is the *time index* and the second one is the *parameter index*.

# Examples

Consider the following example for drawing batches of size 2 for an instance of `DataLoader` constructed with a vector:

```jldoctest
using GeometricMachineLearning
import Random

rng = Random.TaskLocalRNG()
Random.seed!(rng, 123)

dl = DataLoader(rand(rng, 5); autoencoder = true)
batch = Batch(2)

batches = batch(dl)

@assert length(batches) == 3
@assert length.(batches) == [2, 2, 1]
@assert sort(collect(union(batches...))) == [(1, i) for i in 1:5]

true

# output

[ Info: You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters.
true
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
function TransformerLoss(batch::Batch{:Transformer})
    TransformerLoss(batch.seq_length, batch.prediction_window)
end

function Base.iterate(nn::NeuralNetwork, ics, batch::Batch{:Transformer}; n_points = 100)
    iterate(nn, ics; n_points = n_points, prediction_window = batch.prediction_window)
end

@doc raw"""
    Batch(batch_size)

Make an instance of `Batch` for a specific batch size.

This is, among others, used to train neural networks of [`NeuralNetworkIntegrator`](@ref) type (as opposed to [`TransformerIntegrator`](@ref)).
"""
function Batch(batch_size::Int)
    Batch{:FeedForward}(batch_size, 1, 1)
end

@doc raw"""
    Batch(batch_size, seq_length)

Make an instance of `Batch` for a specific batch size and a sequence length.

This is used to train neural networks of [`TransformerIntegrator`](@ref) type.

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
function Batch(batch_size::Int, seq_length::Int, prediction_window::Int = seq_length)
    Batch{:Transformer}(batch_size, seq_length, prediction_window)
end

# The classifier's batch: one draw per sample, with the whole input sequence and the whole output.
# Its fifth type parameter is `Int`, because both lengths come from a loader that holds an output.
function Batch(batch_size::Int,
        dl::DataLoader{T, AT, OT, K, Int}) where {T, AT, OT <: AbstractArray, K}
    Batch(batch_size, dl.input_time_steps, dl.output_time_steps)
end

@doc raw"""
    number_of_batches(dl, batch)

Compute the number of batches.

One epoch draws every start position of every parameter index once, in minibatches of the batch
size, so the count is `cld(n · n_params, batch_size)` with `n` the number of start positions of one
parameter index. No two indices are drawn twice, and the last minibatch holds what is left.

# Examples

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: number_of_batches
import Random

Random.seed!(123)

dat = [1, 2, 3, 4, 5]
dl₁ = DataLoader(dat; autoencoder = false, suppress_info = true) # time series-like
dl₂ = DataLoader(dat; autoencoder = true, suppress_info = true) # autoencoder-like
batch = Batch(3)

batches₁ = batch(dl₁)
batches₂ = batch(dl₂)

@assert number_of_batches(dl₁, batch) == 2
@assert number_of_batches(dl₂, batch) == 2
@assert length.(batches₁) == [3, 1]
@assert length.(batches₂) == [3, 2]
@assert sort(collect(union(batches₁...))) == [(i, 1) for i in 1:4]
@assert sort(collect(union(batches₂...))) == [(1, i) for i in 1:5]

true

# output

true
```

Here we see that in the *autoencoder case* that last minibatch has an additional element.
"""
number_of_batches(dl::DataLoader, batch::Batch) = cld(
    number_of_start_positions(dl, batch) * dl.n_params, batch.batch_size)

# The start positions of one parameter index: the input window must fit the input's time axis, and
# the target must fit its own. The target is the output at offset `0` where the loader holds one,
# the input at offset `seq_length` where it is a time series without an output, and there is none
# where the loader is a set of samples without an output.
function number_of_start_positions(dl::DataLoader{T, AT, OT, K},
        batch::Batch) where {T, AT, OT, K}
    input_window = dl.input_time_steps - batch.seq_length + 1
    target_window = if OT <: AbstractArray
        dl.output_time_steps - batch.prediction_window + 1
    elseif K === :TimeSeries
        input_window - batch.prediction_window
    else
        input_window
    end
    check_window(min(input_window, target_window), dl, batch)
end

# A loader or a batch that leaves no minibatch is a caller's mistake: an epoch of none trains
# nothing and says nothing, so the numbers that decide it are named here. The message is built only
# where it is thrown, so the check itself allocates nothing.
function check_window(n::Int, dl::DataLoader, batch::Batch)
    n ≥ 1 && dl.n_params ≥ 1 || throw(ArgumentError(
        "the `DataLoader` has $(dl.input_time_steps) time steps" *
        (dl.output === nothing ? "" : " and $(dl.output_time_steps) in its output") *
        ", with `n_params = $(dl.n_params)`, and the `Batch` asks for a sequence length of " *
        "$(batch.seq_length) and a prediction window of $(batch.prediction_window), which " *
        "leaves no minibatch."))
    n
end

# The index set of one epoch: the product of the shuffled start positions and parameter indices, cut
# into minibatches of the batch size.
function batch_over_two_axes(batch::Batch, number_columns::Int, third_dim::Int)
    complete_indices = vec(
        collect(Iterators.product(shuffle(1:number_columns), shuffle(1:third_dim))))
    map(collect, Iterators.partition(complete_indices, batch.batch_size))
end

function (batch::Batch)(dl::DataLoader)
    batch_over_two_axes(batch, number_of_start_positions(dl, batch), dl.n_params)
end

@kernel function assign_from_vector_of_tuples_kernel!(gathered, data, indices, offset)
    i, j, k = @index(Global, NTuple)

    gathered[i, j, k] = data[i, indices[1, k] + offset + j - 1, indices[2, k]]
end

# A kernel cannot index a `Vector{Tuple{Int, Int}}`, so the batch indices go in as a `2 × n` matrix:
# the host builds it and the backend takes it in one copy.
function convert_vector_of_tuples_to_matrix(backend::Backend,
        batch_indices_tuple::Vector{Tuple{Int, Int}})
    map_to_new_backend([index[i] for i in 1:2, index in batch_indices_tuple], backend)
end

# The gather kernel, applied to a tensor or to the `q` and `p` of a `(q, p)` pair. The gathered
# tensor holds `width` positions from `offset` on, for every drawn start position and parameter
# index, and has the element type of the data it reads — so one integer target stays integer. The
# first axis is the loader's own `output_dim`/`input_dim`, halved for each of a `(q, p)` pair.
function gather(data::AbstractArray{T, 3}, dim::Int, indices::AbstractArray{Int, 2},
        offset::Int, width::Int, batch_size::Int) where {T}
    backend = networkbackend(data)
    gathered = KernelAbstractions.allocate(backend, T, dim, width, batch_size)
    gather! = assign_from_vector_of_tuples_kernel!(backend)
    gather!(gathered, data, indices, offset, ndrange = size(gathered))
    gathered
end

function gather(data::NamedTuple{(:q, :p), Tuple{AT, AT}}, dim::Int,
        indices::AbstractArray{Int, 2}, offset::Int, width::Int,
        batch_size::Int) where {T, AT <: AbstractArray{T, 3}}
    map(tensor -> gather(tensor, dim ÷ 2, indices, offset, width, batch_size), data)
end

@doc raw"""
    convert_input_and_batch_indices_to_array(dl, batch, batch_indices)

Assign batch data based on (i) input and (ii) batch indices.

# Examples

```jldoctest
using GeometricMachineLearning

dl = DataLoader([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    autoencoder = true, suppress_info = true)
batch = Batch(3)
batch_indices = [(1, 1), (1, 3), (1, 5)]

GeometricMachineLearning.convert_input_and_batch_indices_to_array(dl, batch, batch_indices)

# output

1×1×3 Array{Float64, 3}:
[:, :, 1] =
 0.1

[:, :, 2] =
 0.3

[:, :, 3] =
 0.5
```
"""
function convert_input_and_batch_indices_to_array(
        dl::DataLoader{T, AT, OT, K}, batch::Batch,
        batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, AT, OT, K}
    batch_size = length(batch_indices_tuple)
    indices = convert_vector_of_tuples_to_matrix(networkbackend(dl), batch_indices_tuple)
    input = gather(dl.input, dl.input_dim, indices, 0, batch.seq_length, batch_size)
    # a loader that holds an output is a set of samples whatever its kind: the target is its output;
    # without one, a time series predicts its own continuation and a set of samples has no target
    if dl.output !== nothing
        input,
        gather(dl.output, dl.output_dim, indices, 0, batch.prediction_window, batch_size)
    elseif K === :TimeSeries
        input,
        gather(dl.input, dl.input_dim, indices, batch.seq_length, batch.prediction_window,
            batch_size)
    else
        input
    end
end
