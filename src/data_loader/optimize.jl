function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, CT, Nothing}, batch::Batch, loss::NetworkLoss) where {T, AT<:AbstractArray{T, 3}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}, CT<:Union{AT, BT}}
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

function (o::Optimizer)(nn::NeuralNetwork{<:TransformerIntegrator}, dl::DataLoader, batch::Batch, n_epochs::Int=1)
    loss = TransformerLoss(batch)
    o(nn, dl, batch, n_epochs, loss)
end

function (o::Optimizer)(nn::NeuralNetwork{<:TransformerIntegrator}, dl::DataLoader, batch::Batch{Int}, n_epochs::Int=1)
    loss = TransformerLoss(batch)
end