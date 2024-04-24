"""
Abstract `AutoEncoder` type. If a custom `<:AutoEncoder` architecture is implemented it should have the fields `full_dim`, `reduced_dim`, `n_encoder_blocks` and `n_decoder_blocks`. Further the routines `get_encoder`, `get_decoder`, `get_encoder_parameters` and `get_decoder_parameters` should be extended.
"""
abstract type AutoEncoder <: Architecture end

abstract type Encoder <: Architecture end 

abstract type Decoder <: Architecture end

struct UnknownEncoder <: Encoder end 

struct UnknownDecoder <: Decoder end

#=
"""
Returns a chain that contains the layers of the encoder. 
"""
get_encoder(::AutoEncoder) = error("`get_encoder` not implemented for this type of autoencoder.")
"""
Returns a chain that contains the layers of the decoder. 
"""
get_decoder(::AutoEncoder) = error("`get_decoder` not implemented for this type of autoencoder.")

"""
Returns the parameters of the encoder. 
"""
get_encoder_parameters(::NeuralNetwork{<:AutoEncoder}) = error("`get_encoder_parameters` not implemented for this type of autoencoder.")
"""
Returns the parameters of the decoder. 
"""
get_decoder_parameters(::NeuralNetwork{<:AutoEncoder}) = error("`get_decoder_parameters` not implemented for this type of autoencoder.")
=#

function get_encoder(arch::AutoEncoder)
    encoder_iterations = reverse(compute_iterations(arch.full_dim, arch.reduced_dim, arch.n_encoder_blocks))
    Chain(encoder_layers_from_iteration(arch, encoder_iterations)...)
end

function get_decoder(arch::AutoEncoder)
    decoder_iterations = compute_iterations(arch.full_dim, arch.reduced_dim, arch.n_decoder_blocks)
    Chain(decoder_layers_from_iteration(arch, decoder_iterations)...)
end

function get_encoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_encoder_layers = length(get_encoder(nn.architecture).layers)
    nn.params[1:n_encoder_layers]
end

function get_decoder_parameters(nn::NeuralNetwork{<:AutoEncoder})
    n_decoder_layers = length(get_decoder(nn.architecture).layers)
    nn.params[(end - n_decoder_layers):end]
end

function get_encoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(UnknownEncoder(), get_encoder(nn.architecture), get_encoder_parameters(nn), get_backend(nn))
end

function get_decoder(nn::NeuralNetwork{<:AutoEncoder})
    NeuralNetwork(UnknownDecoder(), get_decoder(nn.architecture), get_decoder_parameters(nn), get_backend(nn))
end