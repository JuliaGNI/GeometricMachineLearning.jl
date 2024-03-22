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

description(::Val{:Batch}) = raw"""
`Batch` is a struct with an associated functor that acts on an instance of `DataLoader`. 

The constructor of `Batch` takes `batch_size` (an integer) as input argument. Optionally we can provide `seq_length` if we deal with time series data and want to draw batches of a certain *length* (i.e. a range contained in the second dimension of the input array).
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

function Batch(batch_size, seq_length = nothing, prediction_window = nothing)
    new{typeof(seq_length)}(batch_size, seq_length, prediction_window)
end

# if no prediction window is provided it is set to 1.
function Batch(batch_size::Int, seq_length::Int, ::Nothing)
    new{typeof(seq_length)}(batch_size, seq_length, 1)
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
function batch_over_one_axis(batch::Batch{Nothing}, number_columns::Int)
    indices = shuffle(1:number_columns)
    n_batches = Int(ceil(number_columns / batch.batch_size))
    batches = ()
    for batch_number in 1:(n_batches - 1)
        batches = (batches..., indices[(batch_number - 1) * batch.batch_size + 1 : batch_number * batch.batch_size])
    end
    # this last line is needed if the number of columns is not divisible by the batch size. 
    (batches..., indices[(n_batches - 1) * batch.batch_size + 1:number_columns])
end

function batch_over_two_axis(batch::Batch{Nothing}, number_columns::Int, third_dim::Int)
    time_indices = shuffle(1:(number_columns - batch.seq_length - batch.prediction_window))
    parameter_indices = shuffle(1:third_dim)
    complete_indices = Iterators.product(time_indices, parameter_indices) |> collect |> vec
    batches = ()
    n_batches = Int(ceil((number_columns - batch.seq_length - batch.prediction_window) * dl.n_params / batch.batch_size))
    for batch_number in 1:(n_batches - 1)
        batches = (batches..., complete_indices[(batch_number - 1) * batch.batch_size + 1 : batch_number * batch.batch_size])
    end
    (batches..., complete_indices[(n_batches - 1) * batch.batch_size + 1:end])
end

(::Batch{Nothing})(::DataLoader{T, AT, Nothing, TimeSteps}) where {T <: Number, AT} = error("Need to provide `seq_length` when dealing with time series data.")

function (batch::Batch{<:Nothing})(dl::DataLoader{T, BT, OT, RegularData}) where {T, AT<:AbstractArray{T}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}, OT}
    batch_over_one_axis(batch, dl.n_params)
end

function (batch::Batch{<:Nothing})(dl::DataLoader{T, BT, OT, TimeSteps}) where {T, AT<:AbstractArray{T}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}, OT}
    batch_over_one_axis(batch, dl.input_time_steps)
end

# Batching for tensor with three axes (unsupervised learning). 
function (batch::Batch{Int})(dl::DataLoader{T, BT, Nothing}) where {T, AT<:AbstractArray{T, 3}, BT<:Union{AT, NamedTuple{(:q, :p), Tuple{AT, AT}}}}
    batch_over_two_axis(batch, dl.input_time_steps, dl.n_params)
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

@doc raw"""
Optimize for an entire epoch. For this you have to supply: 
- an instance of the optimizer.
- the neural network model 
- the parameters of the model 
- the data (in form of `DataLoader`)
- in instance of `Batch` that contains `batch_size` (and optionally `seq_length`)

With the optional argument:
- the loss, which takes the `model`, the parameters `ps` and an instance of `DataLoader` as input.

The output of `optimize_for_one_epoch!` is the average loss over all batches of the epoch:
```math
output = \frac{1}{\mathtt{steps\_per\_epoch}}\sum_{t=1}^\mathtt{steps\_per\_epoch}loss(\theta^{(t-1)}).
```
This is done because any **reverse differentiation** routine always has two outputs: a pullback and the value of the function it is differentiating. In the case of zygote: `loss_value, pullback = Zygote.pullback(ps -> loss(ps), ps)` (if the loss only depends on the parameters).
"""
function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT, BT}, batch::Batch, loss) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    count = 0
    total_error = T(0)
    batches = batch(dl)
    @views for batch_indices in batches 
        count += 1
        # these `copy`s should not be necessary! coming from a Zygote problem!
        input_batch = copy(dl.input[:, :, batch_indices])
        output_batch = copy(dl.output[:, :, batch_indices])
        loss_value, pullback = Zygote.pullback(ps -> loss(model, ps, input_batch, output_batch), ps)
        total_error += loss_value
        dp = pullback(one(loss_value))[1]
        optimization_step!(opt, model, ps, dp)
    end
    total_error / count
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
    i, k = @index(Global, NTuple)

    q_output[i, 1, k] = input.q[i, indices[1, k] + seq_length, indices[2, k]]
    p_output[i, 1, k] = input.p[i, indices[1, k] + seq_length, indices[2, k]]
end

@kernel function assign_output_from_vector_of_tuples_kernel!(output::AT, data::AT, indices::AbstractArray{Int, 2}, seq_length::Int) where {T, AT<:AbstractArray{T, 3}}
    i, k = @index(Global, NTuple)

    output[i, 1, k] = data[i, indices[1, k] + seq_length, indices[2, k]]
end

"""
Takes the output of the batch functor and uses it to create the corresponding array (NamedTuples). 
"""
function convert_input_and_batch_indices_to_array(dl::DataLoader{T, BT}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, AT<:AbstractArray{T, 3}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}
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
    assign_input_from_vector_of_tuples!(q_input, p_input, dl.input, batch_indices, ndrange=(dl.input_dim ÷ 2, batch.seq_length, length(batch_indices_tuple)))

    q_output = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, 1, length(batch_indices_tuple))
    p_output = similar(q_output)

    assign_output_from_vector_of_tuples! = assign_output_from_vector_of_tuples_kernel!(backend)
    assign_output_from_vector_of_tuples!(q_output, p_output, dl.input, batch_indices, batch.seq_length, ndrange=(dl.input_dim ÷ 2, length(batch_indices_tuple)))

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

    output = KernelAbstractions.allocate(backend, T, dl.input_dim, 1, length(batch_indices_tuple))

    assign_output_from_vector_of_tuples! = assign_output_from_vector_of_tuples_kernel!(backend)
    assign_output_from_vector_of_tuples!(output, dl.input, batch_indices, batch.seq_length, ndrange=(dl.input_dim, length(batch_indices_tuple)))

    input, output
end

function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, CT, Nothing}, batch::Batch, loss) where {T, AT<:AbstractArray{T, 3}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}, CT<:Union{AT, BT}}
    count = 0
    total_error = T(0)
    batches = batch(dl)
    @views for batch_indices in batches 
        count += 1
        # these `copy`s should not be necessary! coming from a Zygote problem!
        input_nt, output_nt = convert_input_and_batch_indices_to_array(dl, batch, batch_indices)
        loss_value, pullback = Zygote.pullback(ps -> loss(model, ps, input_nt, output_nt), ps)
        total_error += loss_value
        dp = pullback(one(loss_value))[1]
        optimization_step!(opt, model, ps, dp)
    end
    total_error / count
end

function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT, BT}, batch::Batch) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    optimize_for_one_epoch!(opt, model, ps, dl, batch, loss)
end

function optimize_for_one_epoch!(opt::Optimizer, nn::NeuralNetwork, dl::DataLoader, batch::Batch)
    optimize_for_one_epoch!(opt, nn.model, nn.params, dl, batch)
end

"""
This routine is called if a `DataLoader` storing *symplectic data* (i.e. a `NamedTuple`) is supplied.
"""
function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT}, batch::Batch, loss) where {T, AT<:NamedTuple}
    count = 0 
    total_error = T(0)
    batches = batch(dl)
    @views for batch_indices in batches 
        count += 1
        input_batch = (q=copy(dl.input.q[:,batch_indices]), p=copy(dl.input.p[:,batch_indices]))
        # add +1 here !!!!
        output_batch = (q=copy(dl.input.q[:,batch_indices.+1]), p=copy(dl.input.p[:,batch_indices.+1]))
        loss_value, pullback = Zygote.pullback(ps -> loss(model, ps, input_batch, output_batch), ps)
        total_error += loss_value 
        dp = pullback(one(loss_value))[1]
        optimization_step!(opt, model, ps, dp)
    end
    total_error / count
end

@doc raw"""
A functor for `Optimizer`. It is called with:
    - `nn::NeuralNetwork`
    - `dl::DataLoader`
    - `batch::Batch`
    - `n_epochs::Int`
    - `loss`

The last argument is a function through which `Zygote` differentiates. This argument is optional; if it is not supplied `GeometricMachineLearning` defaults to an appropriate loss for the `DataLoader`.
"""
function (o::Optimizer)(nn::NeuralNetwork, dl::DataLoader, batch::Batch, n_epochs::Int, loss)
    progress_object = ProgressMeter.Progress(n_epochs; enabled=true)
    loss_array = zeros(n_epochs)
    for i in 1:n_epochs
        loss_array[i] = optimize_for_one_epoch!(o, nn.model, nn.params, dl, batch, loss)
        ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_array[i])]) 
    end
    loss_array
end

function (o::Optimizer)(nn::NeuralNetwork, dl::DataLoader, batch::Batch, n_epochs::Int=1)
    o(nn, dl, batch, n_epochs, loss)
end