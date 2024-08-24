@doc raw"""
    NetworkLoss

An abstract type for all the neural network losses. 
If you want to implement `CustomLoss <: NetworkLoss` you need to define a functor:
```julia
(loss::CustomLoss)(model, ps, input, output)
```
where `model` is an instance of an `AbstractExplicitLayer` or a `Chain` and `ps` the parameters.

See [`FeedForwardLoss`](@ref), [`TransformerLoss`](@ref), [`AutoEncoderLoss`](@ref) and [`ReducedLoss`](@ref) for examples.
"""
abstract type NetworkLoss end 

function (loss::NetworkLoss)(nn::NeuralNetwork, input::QPTOAT, output::QPTOAT)
    loss(nn.model, nn.params, input, output)
end

function _compute_loss(output_prediction::QPTOAT, output::QPTOAT)
    _norm(_diff(output_prediction, output)) / _norm(output)
end 

function _compute_loss(model::Union{AbstractExplicitLayer, Chain}, ps::Union{Tuple, NamedTuple}, input::QPTOAT, output::QPTOAT)
    output_prediction = model(input, ps)
    _compute_loss(output_prediction, output)
end

function (loss::NetworkLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::QPTOAT, output::QPTOAT)
    _compute_loss(model, ps, input, output)
end

@doc raw"""
    TransformerLoss(seq_length, prediction_window)

Make an instance of the transformer loss. 

This should be used together with a neural network of type [`TransformerIntegrator`](@ref).

# Example

`TransformerLoss` applies a neural network to an input and compares it to the `output` via an ``L_2`` norm:

```jldoctest 
using GeometricMachineLearning
using LinearAlgebra: norm
import Random

const d = 2
const seq_length = 3
const prediction_window = 2

Random.seed!(123)
arch = StandardTransformerIntegrator(d)
nn = NeuralNetwork(arch)

input_mat =  [1. 2. 3.; 4. 5. 6.]
output_mat = [1. 2.; 3. 4.]
loss = TransformerLoss(seq_length, prediction_window)

# start of prediction
const sop = seq_length - prediction_window + 1
loss(nn, input_mat, output_mat) ≈ norm(output_mat - nn(input_mat)[:, sop:end]) / norm(output_mat)

# output

true
```

So `TransformerLoss` simply does:

```math
    \mathtt{loss}(\mathcal{NN}, \mathtt{input}, \mathtt{output}) = || \mathcal{NN}(\mathtt{input})[(\mathtt{sl} - \mathtt{pw} + 1):\mathtt{end}] - \mathtt{output} || / || \mathtt{output} ||,
```
where ``||\cdot||`` is the ``L_2`` norm. 

# Parameters

The `prediction_window` specifies how many time steps are predicted into the future.
It defaults to the value specified for `seq_length`.
"""
struct TransformerLoss <: NetworkLoss
    seq_length::Int
    prediction_window::Int
end

TransformerLoss(seq_length::Int) = TransformerLoss(seq_length, seq_length)

# This crops the output array of the neural network so that it conforms with the output it should be compared to. This is needed for the transformer loss. 
function crop_array_for_transformer_loss(nn_output::AT, output::BT) where {T, T2, AT <: AbstractArray{T, 3}, BT <: AbstractArray{T2, 3}}
    @view nn_output[axes(output, 1), axes(output, 2) .+ size(nn_output, 2) .- size(output, 2), axes(output, 3)]
end

function (loss::TransformerLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AT, output::AT) where {T, AT <: AbstractArray{T, 3}}
    input_dim, input_seq_length = size(input)
    output_dim, output_prediction_window = size(output)
    @assert input_dim == output_dim
    @assert input_seq_length == loss.seq_length
    @assert output_prediction_window == loss.prediction_window

    predicted_output_uncropped = model(input, ps)
    predicted_output_cropped = crop_array_for_transformer_loss(predicted_output_uncropped, output)
    _compute_loss(predicted_output_cropped, output)
end

function (loss::TransformerLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AT, output::AT) where {T, AT <: AbstractArray{T, 2}}
    loss(model, ps, reshape(input, size(input)..., 1), reshape(output, size(output)..., 1))
end

# @doc raw"""
#     ClassificationTransformerLoss()
# 
# Make an instance of `ClassificationTransformerLoss`.
# 
# This is to be used together with a [`ClassificationTransformer`](@ref).
# 
# It takes an input, parses it to the transformer and then crops it to conform with the desired output size.
# 
# Suppose the input is of dimension ``\mathtt{td}\times\mathtt{sl}``, where `td` is *transformer dimension* and `sl` is *sequence length*.
# The output of the transformer will again be of the same dimension: 
# 
# ```math
# \mathrm{output}\in\mathbb{R}^{\mathtt{td}\times\mathtt{sl}}.
# ```
# 
# if the output dimension `cl` of the [`ClassificationLayer`](@ref) is differnt form `td`.
# """
struct ClassificationTransformerLoss <: NetworkLoss end

function (loss::ClassificationTransformerLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AbstractArray, output::AbstractArray)
    predicted_output_uncropped = model(input, ps)
    # predicted_output_cropped = crop_array_for_transformer_loss(predicted_output_uncropped, output)
    norm(predicted_output_uncropped - output) / norm(output)
end

@doc raw"""
    FeedForwardLoss()

Make an instance of a loss for feedforward neural networks.

This should be used together with a neural network of type [`NeuralNetworkIntegrator`](@ref).

# Example 

`FeedForwardLoss` applies a neural network to an input and compares it to the `output` via an ``L_2`` norm:

```jldoctest 
using GeometricMachineLearning
using LinearAlgebra: norm
import Random
Random.seed!(123)

const d = 2
arch = GSympNet(d)
nn = NeuralNetwork(arch)

input_vec =  [1., 2.]
output_vec = [3., 4.]
loss = FeedForwardLoss()

loss(nn, input_vec, output_vec) ≈ norm(output_vec - nn(input_vec)) / norm(output_vec)

# output

true
```

So `FeedForwardLoss` simply does:

```math
    \mathtt{loss}(\mathcal{NN}, \mathtt{input}, \mathtt{output}) = || \mathcal{NN}(\mathtt{input}) - \mathtt{output} || / || \mathtt{output}||,
```
where ``||\cdot||`` is the ``L_2`` norm. 

# Parameters

This loss does not have any parameters.
"""
struct FeedForwardLoss <: NetworkLoss end

@doc raw"""
    AutoEncoderLoss()

Make an instance of `AutoEncoderLoss`.

This loss should always be used together with a neural network of type [`AutoEncoder`](@ref) (and it is also the default for training such a network). 

# Example

`AutoEncoderLoss` applies a neural network to an input and compares it to the `output` via an ``L_2`` norm:

```jldoctest 
using GeometricMachineLearning
using LinearAlgebra: norm
import Random
Random.seed!(123)

const N = 4
const n = 1
arch = SymplecticAutoencoder(2*N, 2*n)
nn = NeuralNetwork(arch)

input_vec =  [1., 2., 3., 4., 5., 6., 7., 8.]
loss = AutoEncoderLoss()

loss(nn, input_vec) ≈ norm(input_vec - nn(input_vec)) / norm(input_vec)

# output

true
```

So `AutoEncoderLoss` simply does:

```math
    \mathtt{loss}(\mathcal{NN}, \mathtt{input}) = || \mathcal{NN}(\mathtt{input}) - \mathtt{input} || / || \mathtt{input} ||,
```
where ``||\cdot||`` is the ``L_2`` norm. 

# Parameters

This loss does not have any parameters.
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

# Example

`ReducedLoss` applies the *encoder*, *integrator* and *decoder* neural networks in this order to an input and compares it to the `output` via an ``L_2`` norm:

```jldoctest 
using GeometricMachineLearning
using LinearAlgebra: norm
import Random
Random.seed!(123)

const N = 4
const n = 1

Ψᵉ = NeuralNetwork(Chain(Dense(N, n), Dense(n, n))) |> encoder
Ψᵈ = NeuralNetwork(Chain(Dense(n, n), Dense(n, N))) |> decoder
transformer = NeuralNetwork(StandardTransformerIntegrator(n))

input_mat =  [1.  2.;  3.  4.;  5.  6.;  7.  8.]
output_mat = [9. 10.; 11. 12.; 13. 14.; 15. 16.]
loss = ReducedLoss(Ψᵉ, Ψᵈ)

output_prediction = Ψᵈ(transformer(Ψᵉ(input_mat)))
loss(transformer, input_mat, output_mat) ≈ norm(output_mat - output_prediction) / norm(output_mat)

# output

true
```

So the loss computes: 

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

Internally this does:
```julia
ReducedLoss(encoder(autoencoder), decoder(autoencoder))
```
so it calls the functions [`encoder`](@ref) and [`decoder`](@ref).
"""
function ReducedLoss(autoencoder::NeuralNetwork{<:AutoEncoder})
    ReducedLoss(encoder(autoencoder), decoder(autoencoder))
end

function (loss::ReducedLoss)(model::Chain, params::Tuple, input::CT, output::CT) where {CT <: QPTOAT}
    _compute_loss(loss.decoder(model(loss.encoder(input), params)), output)
end