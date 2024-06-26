@doc raw"""
An autoencoder [goodfellow2016deep](@cite) is a neural network consisting of an encoder ``\Psi^e`` and a decoder ``\Psi^d``. In the simplest case they are trained on some data set ``\mathcal{D}`` to reduce the following error: 

```math
||\Psi^d\circ\Psi^e(\mathcal{D}) - \mathcal{D}||,
```

which we call the *reconstruction error* or *autoencoder error* (see the docs for [AutoEncoderLoss](@ref)) and ``||\cdot||`` is some norm.

# Implementation

Abstract `AutoEncoder` type. If a custom `<:AutoEncoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim`, `n_encoder_blocks` and `n_decoder_blocks`. Further the routines `encoder`, `decoder`, `encoder_parameters` and `decoder_parameters` should be extended.
"""
abstract type AutoEncoder <: Architecture end

"""
Abstract `Encoder` type. 

See 

# Implementation

If a custom `<:Encoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim` and `n_encoder_blocks`.
"""
abstract type Encoder <: Architecture end 

"""
Abstract `Decoder` type. 

See 

# Implementation

If a custom `<:Decoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim` and `n_decoder_blocks`.
"""
abstract type Decoder <: Architecture end

abstract type SymplecticCompression <: AutoEncoder end

"""
Abstract `SymplecticEncoder` type. 

See [`Encoder`](@ref) for the super type and [`NonLinearSymplecticEncoder`](@ref) for a derived `struct`.
"""
abstract type SymplecticEncoder <: Encoder end 

"""
Abstract `SymplecticDecoder` type.

See [`Decoder`](@ref) for the super type and [`NonLinearSymplecticDecoder`](@ref) for a derived `struct`.
"""
abstract type SymplecticDecoder <: Decoder end

const SymplecticDimensionChange = Union{SymplecticCompression, SymplecticEncoder, SymplecticDecoder}

struct UnknownEncoder <: Encoder 
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
end 

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

"""
This function gives iterations from the full dimension to the reduced dimension (i.e. the intermediate steps). The iterations are given in ascending order. 
"""
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

"""
Takes as input the autoencoder architecture and a vector of integers specifying the layer dimensions in the encoder. Has to return a tuple of `AbstractExplicitLayer`s.
"""
encoder_layers_from_iteration(::AutoEncoder, ::AbstractVector{<:Integer}) = error("You have to implement `encoder_layers_from_iteration` for this autoencoder architecture!")

"""
Takes as input the autoencoder architecture and a vector of integers specifying the layer dimensions in the decoder. Has to return a tuple of `AbstractExplicitLayer`s.
"""
decoder_layers_from_iteration(::AutoEncoder, ::AbstractVector{<:Integer}) = error("You have to implement `decoder_layers_from_iteration` for this autoencoder architecture!")

function encoder_model(arch::AutoEncoder)
    encoder_iterations = reverse(compute_encoder_iterations(arch))
    Chain(encoder_layers_from_iteration(arch, encoder_iterations)...)
end

function decoder_model(arch::AutoEncoder)
    decoder_iterations = compute_decoder_iterations(arch)
    Chain(decoder_layers_from_iteration(arch, decoder_iterations)...)
end

function encoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_encoder_layers = length(encoder_model(nn.architecture).layers)
    nn.params[1:n_encoder_layers]
end

function decoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_decoder_layers = length(decoder_model(nn.architecture).layers)
    nn.params[(end - (n_decoder_layers - 1)):end]
end

function Chain(arch::AutoEncoder)
    Chain(encoder_model(arch).layers..., decoder_model(arch).layers...)
end

"""
    encoder(nn::NeuralNetwork{<:AutoEncoder})

Obtain the *encoder* from a [`AutoEncoder`](@ref) neural network. 

The input is a neural network and the output is as well.
"""
function encoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(UnknownEncoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), encoder_model(nn.architecture), encoder_parameters(nn), get_backend(nn))
end

"""
    decoder(nn::NeuralNetwork{<:AutoEncoder})

Obtain the *decoder* from a [`AutoEncoder`](@ref) neural network. 

The input is a neural network and the output is as well.
"""
function decoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(UnknownDecoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), decoder_model(nn.architecture), decoder_parameters(nn), get_backend(nn))
end

function encoder(nn::NeuralNetwork{<:SymplecticCompression})
    NeuralNetwork(UnknownSymplecticEncoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), encoder_model(nn.architecture), encoder_parameters(nn), get_backend(nn))
end

function decoder(nn::NeuralNetwork{<:SymplecticCompression})
    NeuralNetwork(UnknownSymplecticDecoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), decoder_model(nn.architecture), decoder_parameters(nn), get_backend(nn))
end