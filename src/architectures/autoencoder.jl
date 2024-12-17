@doc raw"""
    AutoEncoder <: Architecture

The abstract `AutoEncoder` type.

An autoencoder [goodfellow2016deep](@cite) is a neural network consisting of an encoder ``\Psi^e`` and a decoder ``\Psi^d``. In the simplest case they are trained on some data set ``\mathcal{D}`` to reduce the following error: 

```math
||\Psi^d\circ\Psi^e(\mathcal{D}) - \mathcal{D}||,
```

which we call the *reconstruction error*, *projection error* or *autoencoder error* (see the docs for [`AutoEncoderLoss`](@ref)) and ``||\cdot||`` is some norm.

# Implementation

`AutoEncoder` is an abstract type. If a custom `<:AutoEncoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim`, `n_encoder_blocks` and `n_decoder_blocks`. 

`n_encoder_blocks` and `n_decoder_blocks` indicate how often the dimension is changed in the encoder (respectively the decoder).

Further the routines [`encoder`](@ref) and [`decoder`](@ref) should be extended.
"""
abstract type AutoEncoder <: Architecture end

"""
    Encoder <: Architecture

This is the abstract `Encoder` type. 

Most often this should not be called directly, but rather through the [`encoder`](@ref) function.

# Implementation

If a custom `<:Encoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim` and `n_encoder_blocks`.
"""
abstract type Encoder <: Architecture end 

"""
    Decoder <: Architecture

This is the abstract `Decoder` type. 

Most often this should not be called directly, but rather through the [`decoder`](@ref) function.

# Implementation

If a custom `<:Decoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim` and `n_decoder_blocks`.
"""
abstract type Decoder <: Architecture end

abstract type SymplecticCompression <: AutoEncoder end

"""
    SymplecticEncoder <: Encoder

This is the abstract `SymplecticEncoder` type. 

See [`Encoder`](@ref) for the super type and [`NonLinearSymplecticEncoder`](@ref) for a derived `struct`.
"""
abstract type SymplecticEncoder <: Encoder end 

"""
    SymplecticDecoder <: Decoder

This is the abstract `SymplecticDecoder` type.

See [`Decoder`](@ref) for the super type and [`NonLinearSymplecticDecoder`](@ref) for a derived `struct`.
"""
abstract type SymplecticDecoder <: Decoder end

const SymplecticDimensionChange = Union{SymplecticCompression, SymplecticEncoder, SymplecticDecoder}

"""
    UnknownEncoder(full_dim, reduced_dim, n_encoder_blocks)

Make an instance of `UnknownEncoder`.

This should be used if one wants to use an [`Encoder`](@ref) that does not have any specific structure.

# Examples

We show how to make an encoder from a custom architecture:

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: UnknownEncoder, params

model = Chain(Dense(5, 3, tanh; use_bias = false), Dense(3, 2, identity; use_bias = false))
nn = NeuralNetwork(UnknownEncoder(5, 2, 2), model, params(NeuralNetwork(model)), CPU())

typeof(nn) <: NeuralNetwork{<:GeometricMachineLearning.Encoder}

# output

true
```
"""
struct UnknownEncoder <: Encoder 
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
end 

"""
    UnknownDecoder(full_dim, reduced_dim, n_encoder_blocks)

Make an instance of `UnknownDecoder`.

This should be used if one wants to use an [`Decoder`](@ref) that does not have any specific structure.

An example of using this can be constructed analogously to [`UnknownDecoder`](@ref).
"""
struct UnknownDecoder <: Decoder 
    full_dim::Int 
    reduced_dim::Int 
    n_decoder_blocks::Int
end

struct UnknownSymplecticEncoder <: SymplecticEncoder 
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
end 

struct UnknownSymplecticDecoder <: SymplecticDecoder 
    full_dim::Int 
    reduced_dim::Int 
    n_decoder_blocks::Int
end

# """
# This function gives iterations from the full dimension to the reduced dimension (i.e. the intermediate steps). The iterations are given in ascending order. 
# """
function compute_iterations(full_dim::Integer, reduced_dim::Integer, n_blocks::Integer)
    iterations = Vector{Int}(reduced_dim : (full_dim - reduced_dim) ÷ (n_blocks - 1) : full_dim)
    iterations[end] = full_dim
    iterations
end

function compute_encoder_iterations(arch::AutoEncoder)
    compute_iterations(arch.full_dim, arch.reduced_dim, arch.n_encoder_blocks)
end

function compute_decoder_iterations(arch::AutoEncoder)
    compute_iterations(arch.full_dim, arch.reduced_dim, arch.n_decoder_blocks)
end

# """
# Takes as input the autoencoder architecture and a vector of integers specifying the layer dimensions in the encoder. Has to return a tuple of `AbstractExplicitLayer`s.
# """
encoder_layers_from_iteration(::AutoEncoder, ::AbstractVector{<:Integer}) = error("You have to implement `encoder_layers_from_iteration` for this autoencoder architecture!")

# """
# Takes as input the autoencoder architecture and a vector of integers specifying the layer dimensions in the decoder. Has to return a tuple of `AbstractExplicitLayer`s.
# """
decoder_layers_from_iteration(::AutoEncoder, ::AbstractVector{<:Integer}) = error("You have to implement `decoder_layers_from_iteration` for this autoencoder architecture!")

function encoder_model(arch::AutoEncoder)
    encoder_iterations = reverse(compute_encoder_iterations(arch))
    Chain(encoder_layers_from_iteration(arch, encoder_iterations)...)
end

function decoder_model(arch::AutoEncoder)
    decoder_iterations = compute_decoder_iterations(arch)
    Chain(decoder_layers_from_iteration(arch, decoder_iterations)...)
end

# """
#     encoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
# 
# Take a neural network of type [`AutoEncoder`](@ref) and return the parameters of the [`Encoder`](@ref).
# """
function encoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_encoder_layers = length(encoder_model(nn.architecture).layers)
    keys = Tuple(Symbol.(["L$(i)" for i in 1:n_encoder_layers]))
    NeuralNetworkParameters(NamedTuple{keys}(Tuple([params(nn)[key] for key in keys])))
end

# """
#     decoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
# 
# Take a neural network of type [`AutoEncoder`](@ref) and return the parameters of the [`Decoder`](@ref).
# """
function decoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_decoder_layers = length(decoder_model(nn.architecture).layers)
    all_keys = keys(params(nn))
    # "old keys" are the ones describing the correct parameters in params(nn)
    keys_old = Tuple(Symbol.(["L$(i)" for i in (length(all_keys) - (n_decoder_layers - 1)):length(all_keys)]))
    n_keys = length(keys_old)
    # "new keys" are the ones describing the keys in the new NamedTuple
    keys_new = Tuple(Symbol.(["L$(i)" for i in 1:n_keys]))
    NeuralNetworkParameters(NamedTuple{keys_new}(Tuple([params(nn)[key] for key in keys_old])))
end

function Chain(arch::AutoEncoder)
    Chain(encoder_model(arch).layers..., decoder_model(arch).layers...)
end

"""
    encoder(nn::NeuralNetwork{<:AutoEncoder})

Obtain the *encoder* from an [`AutoEncoder`](@ref) neural network. 
"""
function encoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(  UnknownEncoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), 
                    encoder_model(nn.architecture), 
                    encoder_parameters(nn), 
                    networkbackend(nn))
end

function _encoder(nn::NeuralNetwork, full_dim::Integer, reduced_dim::Integer)
    NeuralNetwork(  UnknownEncoder(full_dim, reduced_dim, length(nn.model.layers)), 
                    nn.model, 
                    params(nn), 
                    networkbackend(nn))
end

function input_dimension(::AbstractExplicitLayer{M, N}) where {M, N}
    M
end

function output_dimension(::AbstractExplicitLayer{M, N}) where {M, N}
    N
end

@doc raw"""
    encoder(nn)

Make a neural network of type [`Encoder`](@ref) out of an arbitrary neural network.

# Implementation

Internally this allocates a new nerual network of type [`UnknownEncoder`](@ref) and takes the parameters and the backend from `nn`.
"""
function encoder(nn::NeuralNetwork)
    _encoder(nn, input_dimension(nn.model.layers[1]), output_dimension(nn.model.layers[end]))
end

"""
    decoder(nn::NeuralNetwork{<:AutoEncoder})

Obtain the *decoder* from an [`AutoEncoder`](@ref) neural network.
"""
function decoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(UnknownDecoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), decoder_model(nn.architecture), decoder_parameters(nn), networkbackend(nn))
end

function _decoder(nn::NeuralNetwork, full_dim::Integer, reduced_dim::Integer)
    NeuralNetwork(UnknownDecoder(full_dim, reduced_dim, length(nn.model.layers)), nn.model, params(nn), networkbackend(nn))
end

@doc raw"""
    decoder(nn)

Make a neural network of type [`Decoder`](@ref) out of an arbitrary neural network.

# Implementation

Internally this allocates a new nerual network of type [`UnknownDecoder`](@ref) and takes the parameters and the backend from `nn`.
"""
function decoder(nn::NeuralNetwork)
    _decoder(nn, input_dimension(nn.model.layers[1]), output_dimension(nn.model.layers[end]))
end

function encoder(nn::NeuralNetwork{<:SymplecticCompression})
    NeuralNetwork(UnknownSymplecticEncoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), encoder_model(nn.architecture), encoder_parameters(nn), networkbackend(nn))
end

function decoder(nn::NeuralNetwork{<:SymplecticCompression})
    NeuralNetwork(UnknownSymplecticDecoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), decoder_model(nn.architecture), decoder_parameters(nn), networkbackend(nn))
end