@doc raw"""
    optimize_for_one_epoch!(opt, model, ps, dl, batch, loss, λY)

Sample the data contained in `dl` according to `batch` and optimize for these data.

This step also performs automatic differentiation on `loss`.

`λY` is the *section* for the parameters `ps`.

The output of `optimize_for_one_epoch!` is the average loss over all batches of the epoch:
```math
output = \frac{1}{\mathtt{steps\_per\_epoch}}\sum_{t=1}^\mathtt{steps\_per\_epoch}loss(\theta^{(t-1)}).
```
This is done because any *reverse differentiation* routine always has two outputs: a pullback and the value of the function it is differentiating. In the case of zygote: `loss_value, pullback = Zygote.pullback(ps -> loss(ps), ps)` (if the loss only depends on the parameters).

# Arguments 

The following arguments are mandatory: 
1. an instance of [`Optimizer`](@ref).
2. the neural network model.
3. the neural network parameters `ps`.
4. the data (i.e. an instance of [`DataLoader`](@ref)).
5. in instance of [`Batch`](@ref) that stores `batch_size` (and optionally `seq_length`).
6. `loss::`[`NetworkLoss`](@ref).
7. the *section* `λY` of the parameters `ps`.
"""
function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T}, batch::Batch, loss::Union{typeof(loss), NetworkLoss}, λY) where T
    count = 0
    total_error = T(0)
    batches = batch(dl)
    for batch_indices in batches 
        count += 1
        # these `copy`s should not be necessary! coming from a Zygote problem!
        input_nt_output_nt = convert_input_and_batch_indices_to_array(dl, batch, batch_indices) |> _copy
        loss_value, pullback = if typeof(input_nt_output_nt) <: Tuple
            Zygote.pullback(ps -> loss(model, ps, input_nt_output_nt...), ps)
        else
            Zygote.pullback(ps -> loss(model, ps, input_nt_output_nt), ps)
        end
        total_error += loss_value
        dp = pullback(one(loss_value))[1]
        optimization_step!(opt, λY, ps, dp)
    end
    total_error / count
end

_copy(a::AbstractArray) = copy(a)
_copy(qp::QPT) = (q = copy(qp.q), p = copy(qp.p))
_copy(t::Tuple{<:QPTOAT, <:QPTOAT}) = _copy.(t)

function (o::Optimizer)(nn::NeuralNetwork, dl::DataLoader, batch::Batch, n_epochs::Int, loss::NetworkLoss; show_progress = true)
    Λ = GlobalSection(nn.params)
    progress_object = show_progress == true ? ProgressMeter.Progress(n_epochs; enabled=true) : nothing
    loss_array = zeros(n_epochs)
    for i in 1:n_epochs
        loss_array[i] = optimize_for_one_epoch!(o, nn.model, nn.params, dl, batch, loss, Λ)
        show_progress == true ? ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_array[i])]) : nothing
    end

    loss_array
end

function (o::Optimizer)(nn::NeuralNetwork{<:TransformerIntegrator}, dl::DataLoader, batch::Batch{:Transformer}, n_epochs::Int=1; kwargs...)
    loss = TransformerLoss(batch)
    o(nn, dl, batch, n_epochs, loss; kwargs...)
end

function (o::Optimizer)(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, dl::DataLoader, batch::Batch{:FeedForward}, n_epochs::Int=1; kwargs...)
    loss = FeedForwardLoss()
    o(nn, dl, batch, n_epochs, loss; kwargs...)
end

function (o::Optimizer)(nn::NeuralNetwork{<:AutoEncoder}, dl::DataLoader, batch::Batch{:FeedForward}, n_epochs::Int=1; kwargs...)
    loss = AutoEncoderLoss()
    o(nn, dl, batch, n_epochs, loss; kwargs...)
end