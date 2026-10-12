"""
    train!(nn, opt, dl, batch, n_epochs; loss, show_progress = true)

Train the neural network `nn` with the [`NetworkOptimizer`](@ref) `opt` on the data of the
[`DataLoader`](@ref) `dl` for `n_epochs` epochs, and return the loss history.

Each epoch draws the minibatches of `dl` with the [`Batch`](@ref) `batch` and takes one step per
minibatch: Zygote computes the loss on the minibatch and its gradient, and
`GeometricOptimizers.optimization_step!` writes the new parameters into `params(nn)`, with the
whole gradient, a zero leaf included. Entry `i` of
the history is the mean of the minibatch losses of epoch `i`, each evaluated before its step. The
history has the element type of the loss value, so a `Float64` network on `Float32` data gives a
`Vector{Float64}`.

`loss` is a `NetworkLoss`. Without it, a [`TransformerIntegrator`](@ref) with a
`Batch{:Transformer}` trains on [`TransformerLoss`](@ref)`(batch)`, a
[`NeuralNetworkIntegrator`](@ref) with a `Batch{:FeedForward}` on `FeedForwardLoss()`, and an
autoencoder with a `Batch{:FeedForward}` on [`AutoEncoderLoss`](@ref)`()`; any other network needs
`loss`.

`opt` must be built for `params(nn)`. It keeps the iteration number, the moments and the global
sections, so a second call with the same `opt` resumes the training where the first one stopped;
the only random numbers `train!` draws are those of the minibatch shuffle. `n_epochs < 1` is an
`ArgumentError`, raised before any step.
"""
function train!(nn::NeuralNetwork, opt::NetworkOptimizer, dl::DataLoader, batch::Batch,
        n_epochs::Integer; loss::NetworkLoss = _default_loss(nn.architecture, batch),
        show_progress::Bool = true)
    n_epochs ≥ 1 ||
        throw(ArgumentError("`train!` needs at least one epoch; `n_epochs` is $(n_epochs)."))
    pullback = ZygotePullback(loss)
    progress = ProgressMeter.Progress(n_epochs; enabled = show_progress)
    map(1:n_epochs) do _
        value = _train_epoch!(nn, opt, dl, batch, pullback)
        ProgressMeter.next!(progress; showvalues = [(:TrainingLoss, value)])
        value
    end
end

# The sum takes the type of the first minibatch's loss, so nothing is converted.
function _train_epoch!(nn, opt, dl, batch, pullback)
    batches = batch(dl)
    sum(_train_step!(nn, opt, dl, batch, pullback, indices) for indices in batches) /
    length(batches)
end

function _train_step!(nn, opt, dl, batch, pullback, indices)
    value, back = pullback(params(nn), nn.model,
        convert_input_and_batch_indices_to_array(dl, batch, indices))
    optimization_step!(params(nn), opt, _get_contents(back(one(value))))
    value
end

_default_loss(::TransformerIntegrator, batch::Batch{:Transformer}) = TransformerLoss(batch)
_default_loss(::NeuralNetworkIntegrator, ::Batch{:FeedForward}) = FeedForwardLoss()
_default_loss(arch::AutoEncoder, ::Batch{:FeedForward}) = NetworkLoss(arch)
