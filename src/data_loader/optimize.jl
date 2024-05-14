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
function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T}, batch::Batch, loss::Union{typeof(loss), NetworkLoss}) where T
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

function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, <:Any, Nothing, :RegularData}, batch::Batch, loss::Union{typeof(loss), NetworkLoss}) where T
    count = 0
    total_error = T(0)
    batches = batch(dl)
    @views for batch_indices in batches 
        count += 1
        # these `copy`s should not be necessary! coming from a Zygote problem!
        input_nt = convert_input_and_batch_indices_to_array(dl, batch, batch_indices)
        loss_value, pullback = Zygote.pullback(ps -> loss(model, ps, input_nt), ps)
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
function (o::Optimizer)(nn::NeuralNetwork, dl::DataLoader, batch::Batch, n_epochs::Int, loss::NetworkLoss)
    progress_object = ProgressMeter.Progress(n_epochs; enabled=true)
    loss_array = zeros(n_epochs)
    for i in 1:n_epochs
        loss_array[i] = optimize_for_one_epoch!(o, nn.model, nn.params, dl, batch, loss)
        ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_array[i])]) 
    end

    loss_array
end

function (o::Optimizer)(nn::NeuralNetwork{<:TransformerIntegrator}, dl::DataLoader, batch::Batch{:Transformer}, n_epochs::Int=1)
    loss = TransformerLoss(batch)
    o(nn, dl, batch, n_epochs, loss)
end

function (o::Optimizer)(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, dl::DataLoader, batch::Batch{:FeedForward}, n_epochs::Int=1)
    loss = FeedForwardLoss()
    o(nn, dl, batch, n_epochs, loss)
end

function (o::Optimizer)(nn::NeuralNetwork{<:AutoEncoder}, dl::DataLoader, batch::Batch{:FeedForward}, n_epochs::Int=1)
    loss = AutoEncoderLoss()
    o(nn, dl, batch, n_epochs, loss)
end