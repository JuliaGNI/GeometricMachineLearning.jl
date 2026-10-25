"""
    train!(nn, opt, dl, batch, n_epochs; loss, show_progress = true)

Train the neural network `nn` with the `GeometricOptimizers.TrainingOptimizer` `opt` on the data of
the [`DataLoader`](@ref) or [`ParametricDataLoader`](@ref) `dl` for `n_epochs` epochs, and return
the loss history.

Each epoch draws the minibatches of `dl` with the [`Batch`](@ref) `batch` and takes one step per
minibatch: `loss` computes the loss on the minibatch and its gradient, and
`GeometricOptimizers.optimization_step!` writes the new parameters into `params(nn)`, with the whole
gradient, a zero leaf included. Entry `i` of the history is the mean of the minibatch losses of epoch
`i`, each evaluated before its step. The history has the element type of the loss value, so a
`Float64` network on `Float32` data gives a `Vector{Float64}`.

`loss` is a `NetworkLoss` of AbstractNeuralNetworks or an `AbstractPullback`; a `NetworkLoss` is
trained through a [`ZygotePullback`](@ref). Without it, a [`TransformerIntegrator`](@ref) with a
`Batch{:Transformer}` trains on [`TransformerLoss`](@ref)`(batch)`, a
[`NeuralNetworkIntegrator`](@ref) with a `Batch{:FeedForward}` on `FeedForwardLoss()`, and an
autoencoder with a `Batch{:FeedForward}` on [`AutoEncoderLoss`](@ref)`()`, and any network on a
`ParametricDataLoader` on [`ParametricLoss`](@ref)`()`; any other combination needs `loss`, and
`train!` raises an `ArgumentError` without it.

A network of HamiltonianNeuralNetworks.jl trains on vector-field data with
`loss = SymbolicPullback(arch)` of that package, and on consecutive states with a loss such as its
`SymplecticEulerLoss(arch, timestep)`; see its
[manual](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/latest/).

`opt` is built for `params(nn)`, with `TrainingOptimizer(params(nn); algorithm, linesearch,
retraction)`. It keeps the iteration number, the moments and the global sections, so a second call
with the same `opt` resumes the training where the first one stopped; the only random numbers
`train!` draws are those of the minibatch shuffle. `n_epochs < 1` is an `ArgumentError`, raised
before any step.
"""
function train!(nn::NeuralNetwork, opt::TrainingOptimizer,
        dl::Union{DataLoader, ParametricDataLoader}, batch::Batch, n_epochs::Integer;
        loss::Union{NetworkLoss, AbstractPullback} = default_loss(nn, dl, batch),
        show_progress::Bool = true)
    n_epochs ≥ 1 ||
        throw(ArgumentError("`train!` needs at least one epoch; `n_epochs` is $(n_epochs)."))
    # The pullback is built once per call, not once per step.
    pullback = _as_pullback(loss)
    progress = ProgressMeter.Progress(n_epochs; enabled = show_progress)
    # A loop and not `map(1:n_epochs) do`: the closure around `_train_epoch!` makes the first call
    # of a transformer compile several times as long.
    history = [_train_epoch!(nn, opt, dl, batch, pullback, progress)]
    for _ in 2:n_epochs
        push!(history, _train_epoch!(nn, opt, dl, batch, pullback, progress))
    end
    history
end

# The sum takes the type of the first minibatch's loss, so nothing is converted.
function _train_epoch!(nn, opt, dl, batch, pullback, progress)
    batches = batch(dl)
    total = sum(_train_step!(nn, opt, dl, batch, pullback, indices) for indices in batches)
    value = total / length(batches)
    ProgressMeter.next!(progress; showvalues = [(:TrainingLoss, value)])
    value
end

function _train_step!(nn, opt, dl, batch, pullback, indices)
    value, back = pullback(params(nn), nn.model,
        convert_input_and_batch_indices_to_array(dl, batch, indices))
    optimization_step!(params(nn), opt, back(one(value)))
    value
end

_as_pullback(l::NetworkLoss) = ZygotePullback(l)
_as_pullback(p::AbstractPullback) = p

# A `DataLoader` takes the default loss of the architecture and the batch; a `ParametricDataLoader`
# draws the system parameters of each sample, which `ParametricLoss` hands to the network.
function default_loss(nn::NeuralNetwork, ::DataLoader, batch::Batch)
    NetworkLoss(nn.architecture, batch)
end
default_loss(::NeuralNetwork, ::ParametricDataLoader, ::Batch) = ParametricLoss()

# The default loss of an architecture and a batch. The two-argument methods live here and not beside
# `NetworkLoss(::AutoEncoder)` in `src/loss/losses.jl`, because `Batch` is included after `src/loss/`.
function AbstractNeuralNetworks.NetworkLoss(
        ::TransformerIntegrator, batch::Batch{:Transformer})
    TransformerLoss(batch)
end
function AbstractNeuralNetworks.NetworkLoss(
        arch::Union{NeuralNetworkIntegrator, AutoEncoder}, ::Batch{:FeedForward})
    NetworkLoss(arch)
end
function AbstractNeuralNetworks.NetworkLoss(arch::Architecture, batch::Batch)
    throw(ArgumentError("`train!` has no default loss for a $(nameof(typeof(arch))) with a " *
                        "$(typeof(batch)); pass the loss as the keyword `loss`."))
end
