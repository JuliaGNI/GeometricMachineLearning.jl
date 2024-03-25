abstract type NetworkLoss end 

function (loss::NetworkLoss)(nn::NeuralNetwork, input::AT, output::AT) where {AT <: AbstractArray}
    loss(nn.model, nn.params, input, output)
end

@doc raw"""
The loss for a transformer network (especially a transformer integrator). The constructor is called with:
- `seq_length::Int`
- `prediction_window::Int` (default is 1).
"""
struct TransformerLoss <: NetworkLoss
    seq_length::Int
    prediction_window::Int
end

TransformerLoss(seq_length::Int) = TransformerLoss(seq_length, 1)

@doc raw"""
This crops the output array of the neural network so that it conforms with the output it should be compared to. This is needed for the transformer loss. 
"""
function crop_array_for_transformer_loss(nn_output::AT, output::AT) where {T, AT<:AbstractArray{T, 3}}
    @view nn_output[axes(output, 1), axes(output, 2) .+ size(nn_output, 2) .- size(output, 2), axes(output, 3)]
end

function (loss::TransformerLoss)(model::Chain, ps::Union{Tuple, NamedTuple}, input::AT, output::AT) where {T, AT <: Union{AbstractArray{T, 2}, AbstractArray{T, 3}}}
    input_dim, input_seq_length = size(input)
    output_dim, output_prediction_window = size(output)
    @assert input_dim == output_dim 
    @assert input_seq_length == loss.seq_length
    @assert output_prediction_window == loss.prediction_window

    predicted_output_uncropped = model(input, ps)
    predicted_output_cropped = crop_array_for_transformer_loss(predicted_output_uncropped, output)
    norm(predicted_output_cropped - output) / norm(output)
end

struct FeedForwardLoss <: NetworkLoss end

function (loss::FeedForwardLoss)(model::Chain, ps::Union{Tuple, NamedTuple}, input::AT, output::AT) where {AT <: AbstractArray}
    norm(model(input, ps) - output) / norm(output)
end