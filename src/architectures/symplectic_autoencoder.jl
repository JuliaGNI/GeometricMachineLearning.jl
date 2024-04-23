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

function SymplecticAutoencoder(full_dim::Integer, reduced_dim::Integer; n_encoder_layers::Integer = 4, n_encoder_blocks::Integer = 2, n_decoder_layers::Integer = 1, n_decoder_blocks::Integer = 3, sympnet_upscale::Integer = 5, activation = tanh, encoder_init_q::Bool = true, decoder_init_q_true::Bool = true)
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
This function gives iterations from the full dimension to the reduced dimension (i.e. the intermediate steps). The iterations are given in ascending order. 
"""
function compute_iterations(full_dim::Integer, reduced_dim::Integer, n_blocks::Integer)
    iterations = Vector{Int}(reduced_dim : (full_dim - reduced_dim) ÷ (n_blocks - 1) : full_dim)
    iterations[end] = full_dim 
    iterations
end

function encoder_or_decoder_layers_from_iteration(arch::SymplecticAutoencoder, encoder_iterations::AbstractVector, _determine_layer_type)
    encoder_layers = ()
    encoder_iterations_reduced = encoder_iterations[1:(end - 1)]
    for (i, it) in zip(axes(encoder_iterations_reduced, 1), encoder_iterations_reduced)
        for layer_index in 1:arch.encoder_layers 
            encoder_layers = _determine_layer_type(layer_index) ? (encoder_layers..., GradientQ(it, arch.sympnet_upscale * it, arch.activation)) : (encoder_layers..., GradientP(it, arch.sympnet_upscale * it, arch.activation))
        end
        encoder_layers = (encoder_layers..., PSDLayer(it, encoder_iterations[i + 1]))
    end

    encoder_layers
end

function encoder_layers_from_iteration(arch::SymplecticAutoencoder{:EncoderInitQ}, encoder_iterations::AbstractVector)
    encoder_or_decoder_layers_from_iteration(arch, encoder_iterations, isodd)
end

function encoder_layers_from_iteration(arch::SymplecticAutoencoder{:EncoderInitP}, encoder_iterations::AbstractVector)
    encoder_or_decoder_layers_from_iteration(arch, encoder_iterations, iseven)
end

function decoder_layers_from_iteration(arch::SymplecticAutoencoder{<:Any, :DecoderInitQ}, decoder_iterations::AbstractVector)
    encoder_or_decoder_layers_from_iteration(arch, decoder_iterations, isodd)
end

function decoder_layers_from_iteration(arch::SymplecticAutoencoder{<:Any, :DecoderInitP}, decoder_iterations::AbstractVector)
    encoder_or_decoder_layers_from_iteration(arch, decoder_iterations, iseven)
end

function Chain(arch::SymplecticAutoencoder)
    encoder_iterations = flip(compute_iterations(arch.full_dim, arch.reduced_dim, arch.n_encoder_blocks))
    decoder_iterations = compute_iterations(arch.full_dim, arch.reduced_dim, arch.n_decoder_blocks)

    Chain(encoder_layers_from_iteration(arch, encoder_iterations)..., decoder_layers_from_iteration(arch, decoder_iterations)...)
end