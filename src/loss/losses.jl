@doc raw"""
An abstract type for all the neural network losses. 
If you want to implement `CustomLoss <: NetworkLoss` you need to define a functor:
```julia
    (loss::CustomLoss)(model, ps, input, output)
```
where `model` is an instance of an `AbstractExplicitLayer` or a `Chain` and `ps` the parameters.
"""
abstract type NetworkLoss end 

function (loss::NetworkLoss)(nn::NeuralNetwork, input::CT, output::CT) where {AT<:AbstractArray, BT <: NamedTuple{(:q, :p), Tuple{AT, AT}}, CT <: Union{AT, BT}}
    loss(nn.model, nn.params, input, output)
end

function _compute_loss(output_prediction::CT1, output::CT2) where {AT<:AbstractArray, BT <: NamedTuple{(:q, :p), Tuple{AT, AT}}, CT <: Union{AT, BT}, CT1 <: CT, CT2 <: CT}
    _norm(_diff(output_prediction, output)) / _norm(output)
end 

function _compute_loss(model::Union{AbstractExplicitLayer, Chain}, ps::Union{Tuple, NamedTuple}, input::CT, output::CT) where {AT<:AbstractArray, BT <: NamedTuple{(:q, :p), Tuple{AT, AT}}, CT <: Union{AT, BT}}
    output_prediction = model(input, ps)
    _compute_loss(output_prediction, output)
end

function (loss::NetworkLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::CT, output::CT) where {AT<:AbstractArray, BT <: NamedTuple{(:q, :p), Tuple{AT, AT}}, CT <: Union{AT, BT}}
    _compute_loss(model, ps, input, output)
end

@doc raw"""
    TransformerLoss(seq_length, prediction_window)

Make an instance of the transformer loss. 

This is the loss for a transformer network (especially a transformer integrator). 
    
# Parameters

The `prediction_window` specifies how many time steps are predicted into the future.
It defaults to the value specified for `seq_length`. 
"""
struct TransformerLoss <: NetworkLoss
    seq_length::Int
    prediction_window::Int
end

TransformerLoss(seq_length::Int) = TransformerLoss(seq_length, seq_length)

@doc raw"""
This crops the output array of the neural network so that it conforms with the output it should be compared to. This is needed for the transformer loss. 
"""
function crop_array_for_transformer_loss(nn_output::AT, output::BT) where {T, T2, AT <: AbstractArray{T, 3}, BT <: AbstractArray{T2, 3}}
    @view nn_output[axes(output, 1), axes(output, 2) .+ size(nn_output, 2) .- size(output, 2), axes(output, 3)]
end

function (loss::TransformerLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AT, output::AT) where {T, AT <: Union{AbstractArray{T, 2}, AbstractArray{T, 3}}}
    input_dim, input_seq_length = size(input)
    output_dim, output_prediction_window = size(output)
    @assert input_dim == output_dim 
    @assert input_seq_length == loss.seq_length
    @assert output_prediction_window == loss.prediction_window

    predicted_output_uncropped = model(input, ps)
    predicted_output_cropped = crop_array_for_transformer_loss(predicted_output_uncropped, output)
    _compute_loss(predicted_output_cropped, output)
end

struct ClassificationTransformerLoss <: NetworkLoss end

function (loss::ClassificationTransformerLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AbstractArray, output::AbstractArray)
    predicted_output_uncropped = model(input, ps)
    predicted_output_cropped = crop_array_for_transformer_loss(predicted_output_uncropped, output)
    norm(predicted_output_cropped - output) / norm(output)
end

@doc raw"""
    FeedForwardLoss()

Make an instance of a loss for feedforward neural networks.

This doesn't have any parameters.
"""
struct FeedForwardLoss <: NetworkLoss end

@doc raw"""
This loss should always be used together with a neural network of type [`AutoEncoder`](@ref) (and it is also the default for training such a network). 

It simply computes: 

```math
\mathtt{AutoEncoderLoss}(nn\mathtt{::Loss}, input) = ||nn(input) - input||.
```
"""
struct AutoEncoderLoss <: NetworkLoss end 

function (loss::AutoEncoderLoss)(nn::NeuralNetwork, input)
    loss(nn.model, nn.params, input, input)
end

function (loss::AutoEncoderLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input)
    loss(model, ps, input, input)
end

@doc raw"""
    ReducedLoss(encoder, decoder)

Make an instance of `ReducedLoss` based on an [`Encoder`](@ref) and a [`Decoder`](@ref).

This loss should be used together with a [`NeuralNetworkIntegrator`](@ref) or [`TransformerIntegrator`](@ref).

The loss is computed as: 

```math
\mathrm{loss}_{\mathcal{E}, \mathcal{D}}(\mathcal{NN}, \mathrm{input}, \mathrm{output}) = ||\mathcal{D}(\mathcal{NN}(\mathcal{E}(\mathrm{input}))) - \mathrm{output}||,
```
where ``\mathcal{E}`` is the [`Encoder`](@ref), ``\mathcal{D}`` is the [`Decoder`](@ref).
``\mathcal{NN}`` is the neural network we compute the loss of.
"""
struct ReducedLoss{ET <: NeuralNetwork{<:Encoder}, DT <: NeuralNetwork{<:Decoder}} <: NetworkLoss
    encoder::ET
    decoder::DT
end

"""
    ReducedLoss(autoencoder)

Make an instance of `ReducedLoss` based on a neural network of type [`AutoEncoder`](@ref).
"""
function ReducedLoss(autoencoder::NeuralNetwork{<:AutoEncoder})
    ReducedLoss(encoder(autoencoder), decoder(autoencoder))
end

function (loss::ReducedLoss)(model::Chain, params::Tuple, input::CT, output::CT) where {AT <:AbstractArray, CT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    _compute_loss(loss.decoder(model(loss.encoder(input), params)), output)
end