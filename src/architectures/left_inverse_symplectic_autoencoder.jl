@doc raw"""
    LeftInverseSymplecticAutoencoder(full_dim, reduced_dim; kwargs...)

Make an instance of a left-inverse symplectic autoencoder (left-inverse SAE) for ambient dimension `full_dim` (``2N``) and latent dimension `reduced_dim` (``2n``). Its encoder is an exact left inverse of its decoder, ``\Psi^e \circ \Psi^d = \mathrm{id}_{\mathbb{R}^{2n}}``, for every value of the parameters. The encoder is built from the analytic inverses of the decoder's layers, with the decoder's weights shared ("tied"), and this is what makes it a left inverse.

Unlike [`SymplecticAutoencoder`](@ref), which maintains independent parameter trees for the encoder and decoder, `LeftInverseSymplecticAutoencoder` stores a single parameter tree `ps` for the stage blocks from `reduced_dim` up to `full_dim`:
- [`decoder`](@ref)`(nn)(ξ)` applies the stage blocks in forward order (from `reduced_dim` to `full_dim`), embedding ``\mathbb{R}^{2n} \to \mathbb{R}^{2N}`` via alternating [`GradientLayerQ`](@ref)/[`GradientLayerP`](@ref) layers and Stiefel [`PSDLayer`](@ref) lifts;
- [`encoder`](@ref)`(nn)(z)` applies `reverse(map(inverse_layer, ...))` with the reversed parameter tree, sharing the exact same leaf arrays with zero duplicated parameters and satisfying ``\Psi^e(\Psi^d(\xi)) = \xi`` to machine precision for all parameters `ps`.

# Arguments

Besides the required even integer arguments `full_dim` and `reduced_dim` (`full_dim >= reduced_dim`), the following keyword arguments are supported:
- `n_blocks::Integer = 2`: Sets the spacing of the intermediate dimensions between `reduced_dim` and `full_dim` (`n_blocks >= 1`, and `n_blocks <= (full_dim - reduced_dim) ÷ 2 + 1` when `full_dim > reduced_dim`; a larger value raises an `ArgumentError` when the network is built). The decoder has one stage block per pair of adjacent dimensions, and their number can differ from `n_blocks`: `LeftInverseSymplecticAutoencoder(12, 2; n_blocks = 4)` has five stage blocks, and `LeftInverseSymplecticAutoencoder(6, 4; n_blocks = 2)` has one.
- `n_layers::Integer = 3`: Number of gradient layers preceding each stage `PSDLayer`.
- `n_ambient_layers::Integer = 2`: Number of ambient gradient layers at `full_dim` following the final `PSDLayer`.
- `sympnet_upscale::Integer = 5`: Upscaling factor of each [`GradientLayer`](@ref).
- `activation = tanh`: Activation function in the gradient layers.
- `init_q::Bool = true`: Whether the first gradient layer in each block is of `Q` type.
"""
struct LeftInverseSymplecticAutoencoder{Init, AT} <: SymplecticCompression
    full_dim::Int
    reduced_dim::Int
    n_blocks::Int
    n_layers::Int
    n_ambient_layers::Int
    sympnet_upscale::Int
    activation::AT
end

function LeftInverseSymplecticAutoencoder(
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

    LeftInverseSymplecticAutoencoder{init_q ? :InitQ : :InitP, typeof(activation)}(
        Int(full_dim), Int(reduced_dim), Int(n_blocks), Int(n_layers),
        Int(n_ambient_layers), Int(sympnet_upscale), activation)
end

function compute_decoder_iterations(arch::LeftInverseSymplecticAutoencoder)
    compute_iterations_for_symplectic_system(arch.full_dim, arch.reduced_dim, arch.n_blocks)
end

function _left_inverse_decoder_layers(
        arch::LeftInverseSymplecticAutoencoder,
        decoder_iterations::AbstractVector{<:Integer},
        _determine_layer_type)
    layers = encoder_or_decoder_layers_from_iteration(
        arch, decoder_iterations, arch.n_layers, _determine_layer_type)
    _append_decoder_output_layers(arch, layers, arch.n_ambient_layers, _determine_layer_type)
end

function decoder_layers_from_iteration(arch::LeftInverseSymplecticAutoencoder{:InitQ},
        decoder_iterations::AbstractVector{<:Integer})
    _left_inverse_decoder_layers(arch, decoder_iterations, isodd)
end

function decoder_layers_from_iteration(arch::LeftInverseSymplecticAutoencoder{:InitP},
        decoder_iterations::AbstractVector{<:Integer})
    _left_inverse_decoder_layers(arch, decoder_iterations, iseven)
end

# The model is a `LeftInverseAutoencoderLayer`, whose encoder and decoder share one parameter tree; a
# `Chain` of the two halves would give each half its own parameters.
function Chain(::LeftInverseSymplecticAutoencoder)
    throw(ArgumentError("A LeftInverseSymplecticAutoencoder has no Chain of untied layers: build it with NeuralNetwork(arch) and take encoder(nn) and decoder(nn)."))
end

_reverse_tuple(::Tuple{}) = ()
_reverse_tuple(t::Tuple) = (_reverse_tuple(Base.tail(t))..., first(t))

function _reverse_params(ps::NamedTuple{names}) where {names}
    NamedTuple{names}(_reverse_tuple(values(ps)))
end

function _reverse_params(ps::NetworkParameters)
    NetworkParameters(NamedTuple{keys(ps)}(_reverse_tuple(values(ps))))
end

struct LeftInverseAutoencoderLayer{M, ET <: Tuple, DT <: Tuple} <:
       AbstractExplicitLayer{M, M}
    encoder_layers::ET
    decoder_layers::DT
end

function LeftInverseAutoencoderLayer(arch::LeftInverseSymplecticAutoencoder)
    dec_layers = decoder_layers_from_iteration(arch, compute_decoder_iterations(arch))
    enc_layers = reverse(map(inverse_layer, dec_layers))
    M = arch.full_dim
    ET, DT = typeof(enc_layers), typeof(dec_layers)
    LeftInverseAutoencoderLayer{M, ET, DT}(enc_layers, dec_layers)
end

function parameterlength(l::LeftInverseAutoencoderLayer)
    parameterlength(Chain(l.decoder_layers...))
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        l::LeftInverseAutoencoderLayer, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T}
    initialparameters(rng, initializer, Chain(l.decoder_layers...), backend, T; kwargs...)
end

function (l::LeftInverseAutoencoderLayer)(z, ps::Union{NamedTuple, NetworkParameters})
    ξ = Chain(l.encoder_layers...)(z, _reverse_params(ps))
    Chain(l.decoder_layers...)(ξ, ps)
end

function NeuralNetwork(
        arch::LeftInverseSymplecticAutoencoder, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(arch, LeftInverseAutoencoderLayer(arch), backend, T; kwargs...)
end

function decoder_parameters(nn::NeuralNetwork{<:LeftInverseSymplecticAutoencoder})
    params(nn)
end

function encoder_parameters(nn::NeuralNetwork{<:LeftInverseSymplecticAutoencoder})
    _reverse_params(params(nn))
end

function encoder(nn::NeuralNetwork{<:LeftInverseSymplecticAutoencoder})
    a = nn.architecture
    arch = NonLinearSymplecticEncoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.sympnet_upscale, a.activation)
    model = Chain(nn.model.encoder_layers...)
    NeuralNetwork(arch, model, encoder_parameters(nn), networkbackend(nn))
end

function decoder(nn::NeuralNetwork{<:LeftInverseSymplecticAutoencoder})
    a = nn.architecture
    arch = NonLinearSymplecticDecoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.n_ambient_layers,
        a.sympnet_upscale, a.activation)
    model = Chain(nn.model.decoder_layers...)
    NeuralNetwork(arch, model, decoder_parameters(nn), networkbackend(nn))
end
