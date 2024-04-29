"""
Abstract `AutoEncoder` type. If a custom `<:AutoEncoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim`, `n_encoder_blocks` and `n_decoder_blocks`. Further the routines `get_encoder`, `get_decoder`, `get_encoder_parameters` and `get_decoder_parameters` should be extended.
"""
abstract type AutoEncoder <: Architecture end

"""
Abstract `Encoder` type. If a custom `<:Encoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim` and `n_encoder_blocks`.
"""
abstract type Encoder <: Architecture end 

"""
Abstract `Decoder` type. If a custom `<:Decoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim` and `n_decoder_blocks`.
"""
abstract type Decoder <: Architecture end

abstract type SymplecticCompression <: AutoEncoder end

abstract type SymplecticEncoder <: Encoder end 

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

function get_encoder_model(arch::AutoEncoder)
    encoder_iterations = reverse(compute_encoder_iterations(arch))
    Chain(encoder_layers_from_iteration(arch, encoder_iterations)...)
end

function get_decoder_model(arch::AutoEncoder)
    decoder_iterations = compute_decoder_iterations(arch)
    Chain(decoder_layers_from_iteration(arch, decoder_iterations)...)
end

function get_encoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_encoder_layers = length(get_encoder_model(nn.architecture).layers)
    nn.params[1:n_encoder_layers]
end

function get_decoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_decoder_layers = length(get_decoder_model(nn.architecture).layers)
    nn.params[(end - (n_decoder_layers - 1)):end]
end

function Chain(arch::AutoEncoder)
    Chain(get_encoder_model(arch).layers..., get_decoder_model(arch).layers...)
end

function get_encoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(UnknownEncoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), get_encoder_model(nn.architecture), get_encoder_parameters(nn), get_backend(nn))
end

function get_decoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(UnknownDecoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), get_decoder_model(nn.architecture), get_decoder_parameters(nn), get_backend(nn))
end

function get_encoder(nn::NeuralNetwork{<:SymplecticCompression})
    NeuralNetwork(UnknownSymplecticEncoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), get_encoder_model(nn.architecture), get_encoder_parameters(nn), get_backend(nn))
end

function get_decoder(nn::NeuralNetwork{<:SymplecticCompression})
    NeuralNetwork(UnknownSymplecticDecoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_blocks), get_decoder_model(nn.architecture), get_decoder_parameters(nn), get_backend(nn))
end

function (nn::NeuralNetwork{<:SymplecticDimensionChange})(q::AT, p::AT) where {AT <: AbstractArray}
    nn_applied = nn((q = q, p = p))
    nn_applied.q, nn_applied.p
end