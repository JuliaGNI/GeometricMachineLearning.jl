@doc raw"""
    WeightTiedSymplecticAutoencoder(full_dim, reduced_dim; kwargs...)

Make an instance of a weight-tied symplectic autoencoder for ambient dimension `full_dim` (``2N``) and latent dimension `reduced_dim` (``2n``).

Unlike [`SymplecticAutoencoder`](@ref), which maintains independent parameter trees for the encoder and decoder, `WeightTiedSymplecticAutoencoder` stores a single parameter tree `ps` for the stage blocks from `reduced_dim` up to `full_dim`:
- [`decoder`](@ref)`(nn)(ξ)` applies the stage blocks in forward order (from `reduced_dim` to `full_dim`), embedding ``\mathbb{R}^{2n} \to \mathbb{R}^{2N}`` via alternating [`GradientLayerQ`](@ref)/[`GradientLayerP`](@ref) layers and Stiefel [`PSDLayer`](@ref) lifts;
- [`encoder`](@ref)`(nn)(z)` applies `reverse(map(inverse_layer, ...))` with the reversed parameter tree, sharing the exact same leaf arrays with zero duplicated parameters and satisfying ``\Psi^e(\Psi^d(\xi)) = \xi`` to machine precision for all parameters `ps`.

# Arguments

Besides the required even integer arguments `full_dim` and `reduced_dim` (`full_dim >= reduced_dim`), the following keyword arguments are supported:
- `n_blocks::Integer = 2`: Number of stage blocks between `reduced_dim` and `full_dim` (`n_blocks >= 1`).
- `n_layers::Integer = 3`: Number of gradient layers preceding each stage `PSDLayer`.
- `n_ambient_layers::Integer = 2`: Number of ambient gradient layers at `full_dim` following the final `PSDLayer`.
- `sympnet_upscale::Integer = 5`: Upscaling factor of each [`GradientLayer`](@ref).
- `activation = tanh`: Activation function in the gradient layers.
- `init_q::Bool = true`: Whether the first gradient layer in each block is of `Q` type.
"""
struct WeightTiedSymplecticAutoencoder{Init, AT} <: SymplecticCompression
    full_dim::Int
    reduced_dim::Int
    n_blocks::Int
    n_layers::Int
    n_ambient_layers::Int
    sympnet_upscale::Int
    activation::AT
end

function WeightTiedSymplecticAutoencoder(
        full_dim::Integer, reduced_dim::Integer; n_blocks::Integer = 2,
        n_layers::Integer = 3,
        n_ambient_layers::Integer = 2,
        sympnet_upscale::Integer = 5,
        activation = tanh,
        init_q::Bool = true)
    (full_dim > 0 && reduced_dim > 0 && iseven(full_dim) && iseven(reduced_dim)) ||
        throw(ArgumentError("full_dim ($full_dim) and reduced_dim ($reduced_dim) must be positive even integers."))
    full_dim >= reduced_dim ||
        throw(ArgumentError("full_dim ($full_dim) must be greater than or equal to reduced_dim ($reduced_dim)."))
    n_blocks >= 1 ||
        throw(ArgumentError("n_blocks ($n_blocks) must be at least 1."))
    (n_layers >= 0 && n_ambient_layers >= 0 && sympnet_upscale >= 1) ||
        throw(ArgumentError("n_layers ($n_layers) and n_ambient_layers ($n_ambient_layers) must be non-negative and sympnet_upscale ($sympnet_upscale) must be at least 1."))

    WeightTiedSymplecticAutoencoder{init_q ? :InitQ : :InitP, typeof(activation)}(
        Int(full_dim), Int(reduced_dim), Int(n_blocks), Int(n_layers),
        Int(n_ambient_layers), Int(sympnet_upscale), activation)
end

function compute_decoder_iterations(arch::WeightTiedSymplecticAutoencoder)
    compute_iterations_for_symplectic_system(arch.full_dim, arch.reduced_dim, arch.n_blocks)
end

function compute_encoder_iterations(arch::WeightTiedSymplecticAutoencoder)
    compute_decoder_iterations(arch)
end

function _weight_tied_decoder_layers(
        arch::WeightTiedSymplecticAutoencoder,
        decoder_iterations::AbstractVector{<:Integer},
        _determine_layer_type)
    layers = ()
    decoder_iterations_reduced = decoder_iterations[1:(end - 1)]
    for (i, it) in zip(axes(decoder_iterations_reduced, 1), decoder_iterations_reduced)
        u = arch.sympnet_upscale * it
        for layer_index in 1:arch.n_layers
            if _determine_layer_type(layer_index)
                layers = (layers..., GradientLayerQ(it, u, arch.activation))
            else
                layers = (layers..., GradientLayerP(it, u, arch.activation))
            end
        end
        layers = (layers..., PSDLayer(it, decoder_iterations[i + 1]))
    end
    d_full = arch.full_dim
    u_full = arch.sympnet_upscale * d_full
    for layer_index in 1:arch.n_ambient_layers
        if _determine_layer_type(layer_index)
            layers = (layers..., GradientLayerQ(d_full, u_full, arch.activation))
        else
            layers = (layers..., GradientLayerP(d_full, u_full, arch.activation))
        end
    end
    layers
end

function decoder_layers_from_iteration(arch::WeightTiedSymplecticAutoencoder{:InitQ},
        decoder_iterations::AbstractVector{<:Integer})
    _weight_tied_decoder_layers(arch, decoder_iterations, isodd)
end

function decoder_layers_from_iteration(arch::WeightTiedSymplecticAutoencoder{:InitP},
        decoder_iterations::AbstractVector{<:Integer})
    _weight_tied_decoder_layers(arch, decoder_iterations, iseven)
end

function encoder_layers_from_iteration(arch::WeightTiedSymplecticAutoencoder,
        decoder_iterations::AbstractVector{<:Integer})
    reverse(map(inverse_layer, decoder_layers_from_iteration(arch, decoder_iterations)))
end

function encoder_model(arch::WeightTiedSymplecticAutoencoder)
    decoder_iterations = compute_decoder_iterations(arch)
    Chain(encoder_layers_from_iteration(arch, decoder_iterations)...)
end

function decoder_model(arch::WeightTiedSymplecticAutoencoder)
    decoder_iterations = compute_decoder_iterations(arch)
    Chain(decoder_layers_from_iteration(arch, decoder_iterations)...)
end

_reverse_tuple(::Tuple{}) = ()
_reverse_tuple(t::Tuple) = (_reverse_tuple(Base.tail(t))..., first(t))

function _reverse_params(ps::NamedTuple{names}) where {names}
    NamedTuple{names}(_reverse_tuple(values(ps)))
end

function _reverse_params(ps::NetworkParameters)
    NetworkParameters(NamedTuple{keys(ps)}(_reverse_tuple(values(ps))))
end

struct WeightTiedAutoencoderLayer{M, ET <: Tuple, DT <: Tuple, LT <: Tuple} <:
       AbstractExplicitLayer{M, M}
    encoder_layers::ET
    decoder_layers::DT
    layers::LT
end

function WeightTiedAutoencoderLayer(arch::WeightTiedSymplecticAutoencoder)
    dec_layers = decoder_layers_from_iteration(arch, compute_decoder_iterations(arch))
    enc_layers = reverse(map(inverse_layer, dec_layers))
    all_layers = (enc_layers..., dec_layers...)
    M = arch.full_dim
    ET, DT, LT = typeof(enc_layers), typeof(dec_layers), typeof(all_layers)
    WeightTiedAutoencoderLayer{M, ET, DT, LT}(enc_layers, dec_layers, all_layers)
end

function parameterlength(l::WeightTiedAutoencoderLayer)
    parameterlength(Chain(l.decoder_layers...))
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        l::WeightTiedAutoencoderLayer, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T}
    initialparameters(rng, initializer, Chain(l.decoder_layers...), backend, T; kwargs...)
end

function (l::WeightTiedAutoencoderLayer)(z, ps::Union{NamedTuple, NetworkParameters})
    ξ = Chain(l.encoder_layers...)(z, _reverse_params(ps))
    Chain(l.decoder_layers...)(ξ, ps)
end

function NeuralNetwork(
        arch::WeightTiedSymplecticAutoencoder, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(arch, WeightTiedAutoencoderLayer(arch), backend, T; kwargs...)
end

function decoder_parameters(nn::NeuralNetwork{<:WeightTiedSymplecticAutoencoder})
    params(nn)
end

function encoder_parameters(nn::NeuralNetwork{<:WeightTiedSymplecticAutoencoder})
    _reverse_params(params(nn))
end

function encoder(nn::NeuralNetwork{<:WeightTiedSymplecticAutoencoder})
    a = nn.architecture
    arch = NonLinearSymplecticEncoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.sympnet_upscale, a.activation)
    model = Chain(nn.model.encoder_layers...)
    NeuralNetwork(arch, model, encoder_parameters(nn), networkbackend(nn))
end

function decoder(nn::NeuralNetwork{<:WeightTiedSymplecticAutoencoder})
    a = nn.architecture
    arch = NonLinearSymplecticDecoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.n_ambient_layers,
        a.sympnet_upscale, a.activation)
    model = Chain(nn.model.decoder_layers...)
    NeuralNetwork(arch, model, decoder_parameters(nn), networkbackend(nn))
end
