struct SymplecticAutoencoder{EncoderInit, DecoderInit, AT} <: AutoEncoder 
    full_dim::Int
    reduced_dim::Int 
    n_encoder_layers::Int
    n_encoder_blocks::Int
    n_decoder_layers::Int
    n_decoder_blocks::Int
    sympnet_upscale::Int
    activation::AT
end

struct SymplecticEncoder{AT} <: Encoder
    full_dim::Int
    reduced_dim::Int 
    n_encoder_layers::Int 
    n_encoder_blocks::Int 
    sympnet_upscale::Int 
    activation::AT
end

struct SymplecticDecoder{AT} <: Encoder
    full_dim::Int
    reduced_dim::Int 
    n_decoder_layers::Int 
    n_decoder_blocks::Int 
    sympnet_upscale::Int 
    activation::AT
end

function SymplecticAutoencoder(full_dim::Integer, reduced_dim::Integer; n_encoder_layers::Integer = 4, n_encoder_blocks::Integer = 2, n_decoder_layers::Integer = 1, n_decoder_blocks::Integer = 3, sympnet_upscale::Integer = 5, activation = tanh, encoder_init_q::Bool = true, decoder_init_q::Bool = true)
    @assert full_dim ≥ reduced_dim "The dimension of the full-order model hast to be larger than the dimension of the reduced order model!"
    @assert iseven(full_dim) && iseven(reduced_dim) "The full-order model and the reduced-order model need to be even dimensional!"
    
    if encoder_init_q && decoder_init_q
        SymplecticAutoencoder{:EncoderInitQ, :DecoderInitQ, typeof(activation)}(full_dim, reduced_dim, n_encoder_layers, n_encoder_blocks, n_decoder_layers, n_decoder_blocks, sympnet_upscale, activation)
    elseif encoder_init_q && !decoder_init_q
        SymplecticAutoencoder{:EncoderInitQ, :DecoderInitP, typeof(activation)}(full_dim, reduced_dim, n_encoder_layers, n_encoder_blocks, n_decoder_layers, n_decoder_blocks, sympnet_upscale, activation)
    elseif !encoder_init_q && decoder_init_q
        SymplecticAutoencoder{:EncoderInitP, :DecoderInitQ, typeof(activation)}(full_dim, reduced_dim, n_encoder_layers, n_encoder_blocks, n_decoder_layers, n_decoder_blocks, sympnet_upscale, activation)
    elseif !encoder_init_q && !decoder_init_q
        SymplecticAutoencoder{:EncoderInitP, :DecoderInitP, typeof(activation)}(full_dim, reduced_dim, n_encoder_layers, n_encoder_blocks, n_decoder_layers, n_decoder_blocks, sympnet_upscale, activation)
    end
end

"""
This function gives iterations from the full dimension to the reduced dimension (i.e. the intermediate steps). The iterations are given in ascending order. Only even steps are allowed here.
"""
function compute_iterations_for_symplectic_system(full_dim::Integer, reduced_dim::Integer, n_blocks::Integer)
    full_dim2 = full_dim ÷ 2 
    reduced_dim2 = reduced_dim ÷ 2
    iterations = Vector{Int}(reduced_dim2 : (full_dim2 - reduced_dim2) ÷ (n_blocks - 1) : full_dim2)
    iterations[end] = full_dim2
    iterations * 2
end

function compute_encoder_iterations(arch::SymplecticAutoencoder)
    compute_iterations_for_symplectic_system(arch.full_dim, arch.reduced_dim, arch.n_encoder_blocks)
end

function compute_decoder_iterations(arch::SymplecticAutoencoder)
    compute_iterations_for_symplectic_system(arch.full_dim, arch.reduced_dim, arch.n_decoder_blocks)
end

function encoder_or_decoder_layers_from_iteration(arch::SymplecticAutoencoder, encoder_iterations::AbstractVector{<:Integer}, n_encoder_layers::Integer, _determine_layer_type)
    encoder_layers = ()
    encoder_iterations_reduced = encoder_iterations[1:(end - 1)]
    for (i, it) in zip(axes(encoder_iterations_reduced, 1), encoder_iterations_reduced)
        for layer_index in 1:n_encoder_layers 
            encoder_layers = _determine_layer_type(layer_index) ? (encoder_layers..., GradientLayerQ(it, arch.sympnet_upscale * it, arch.activation)) : (encoder_layers..., GradientLayerP(it, arch.sympnet_upscale * it, arch.activation))
        end
        encoder_layers = (encoder_layers..., PSDLayer(it, encoder_iterations[i + 1]))
    end

    encoder_layers
end

function encoder_layers_from_iteration(arch::SymplecticAutoencoder{:EncoderInitQ}, encoder_iterations::AbstractVector{<:Integer})
    encoder_or_decoder_layers_from_iteration(arch, encoder_iterations, arch.n_encoder_layers, isodd)
end

function encoder_layers_from_iteration(arch::SymplecticAutoencoder{:EncoderInitP}, encoder_iterations::AbstractVector{<:Integer})
    encoder_or_decoder_layers_from_iteration(arch, encoder_iterations, arch.n_encoder_layers, iseven)
end

function decoder_layers_from_iteration(arch::SymplecticAutoencoder{<:Any, :DecoderInitQ}, decoder_iterations::AbstractVector{<:Integer})
    encoder_or_decoder_layers_from_iteration(arch, decoder_iterations, arch.n_decoder_layers, isodd)
end

function decoder_layers_from_iteration(arch::SymplecticAutoencoder{<:Any, :DecoderInitP}, decoder_iterations::AbstractVector{<:Integer})
    encoder_or_decoder_layers_from_iteration(arch, decoder_iterations, arch.n_decoder_layers, iseven)
end

function get_encoder(nn::NeuralNetwork{<:SymplecticAutoencoder})
    arch = SymplecticEncoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_encoder_layers, nn.architecture.n_encoder_blocks, nn.architecture.sympnet_upscale, nn.architecture.activation)
    NeuralNetwork(arch, get_encoder_model(nn.architecture), get_encoder_parameters(nn), get_backend(nn))
end

function get_decoder(nn::NeuralNetwork{<:SymplecticAutoencoder})
    arch = SymplecticDecoder(nn.architecture.full_dim, nn.architecture.reduced_dim, nn.architecture.n_decoder_layers, nn.architecture.n_decoder_blocks, nn.architecture.sympnet_upscale, nn.architecture.activation)
    NeuralNetwork(arch, get_decoder_model(nn.architecture), get_decoder_parameters(nn), get_backend(nn))
end