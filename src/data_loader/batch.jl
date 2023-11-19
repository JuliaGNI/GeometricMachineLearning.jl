@doc raw"""
`Batch` is a struct with an associated functor that acts on `DataLoader`. 

The functor returns indices that are then used in the optimization step (always for an entire epoch).
"""
struct Batch{seq_type <: Union{Nothing, Integer}}
    batch_size::Int
    seq_length::seq_type

    function Batch(batch_size, seq_length = nothing)
        new{typeof(seq_length)}(batch_size, seq_length)
    end
end

hasseqlength(::Batch{<:Integer}) = true
hasseqlength(::Batch{<:Nothing}) = false

function (batch::Batch{<:Nothing})(dl::DataLoader{T, AT}) where {T, AT<:AbstractArray{T, 3}}
    indices = shuffle(1:dl.n_params)
    n_batches = Int(ceil(dl.n_params/batch.batch_size))
    batches = ()
    for batch_number in 1:(n_batches-1)
        batches = (batches..., indices[(batch_number-1)*batch.batch_size + 1:batch_number*batch.batch_size])
    end

    # this is needed because the last batch may not have the full size
    batches = (batches..., indices[( (n_batches-1) * batch.batch_size + 1 ):end])
    batches
end

function (batch::Batch{<:Nothing})(dl::DataLoader{T, AT}) where {T, BT<:AbstractMatrix{T} ,AT<:Union{BT, NamedTuple{(:q, :p), Tuple{BT, BT}}}}
    indices = shuffle(1:dl.input_time_steps)
    n_batches = Int(ceil((dl.input_time_steps-1)/batch.batch_size))
    batches = ()
    for batch_number in 1:(n_batches-1)
        batches = (batches..., indices[(batch_number-1)*batch.batch_size + 1:batch_number*batch.batch_size])
    end
end

(batch::Batch)(dl::NamedTuple) = batch(dl.q)

#=
function (batch::Batch{<:Integer})(dl::DataLoader{T, AT, Nothing}) where {T, AT<:AbstractArray{T, 3}}
    n_starting_points = n_params
    ...
end 
=#

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
output = \frac{1}{mathtt{steps\_per\_epoch}}\sum_{t=1}^mathtt{steps\_per\_epoch}loss(\theta^{(t-1)}).
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
    total_error/count
end

function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT, BT}, batch::Batch) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    optimize_for_one_epoch!(opt, model, ps, dl, batch, loss)
end

"""
TODO: make this work for higher dimensions!!!!
"""
function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT}, batch::Batch, loss) where AT<:NamedTuple
    count = 0 
    total_error = T(0)
    batches = batch(dl)
    @views for batch_indices in batches 
        input_batch = (q=copy(dl.q.input[batch_indices]), p=copy(dl.p.input[batch_indices]))
        # add +1 here !!!!
        output_batch = (q=copy(dl.q.input[batch_indices .+ 1]), p=copy(dl.p.input[batch_indices .+ 1]))
        loss_value, pullback = Zygote.pullback(ps -> loss(model, ps, input_batch, output_batch), ps)
        total_error += loss_value 
        dp = pullback(one(loss_value))[1]
        optimization_step!(opt, model, ps, dp)
    end
    total_error/count
end


function (o::Optimizer)(nn::NeuralNetwork, dl::NamedTuple, batch::Batch, n_epochs::Int, loss)
    loss_array = zeros(n_epochs)
    for i in 1:n_epochs
        loss_array[i] = optimize_for_one_epoch!(o, nn.model, nn.params, dl, batch, loss)
    end
    loss_array
end

function (o::Optimizer)(nn::NeuralNetwork, dl::NamedTuple, batch::Batch, n_epochs::Int)
    o(nn, dl, batch, n_epochs, loss)
end