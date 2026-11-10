@doc raw"""
    EquivariantSymplecticAutoencoder(M, d, d_latent; kwargs...)

Make an instance of a weight-tied, permutation-equivariant symplectic autoencoder for `M` particles in ``d`` spatial dimensions (ambient dimension ``2N = 2Md``) with a latent space of `M` latent particles in ``\mathbb{R}^{\tilde d}``, ``\tilde d`` = `d_latent` (latent dimension ``2n = 2M\tilde d``).

Every layer acts on the particles with weights shared by all of them. The decoder alternates blocks of pairwise shears ([`PairwiseSymplecticLayerQ`](@ref)/[`PairwiseSymplecticLayerP`](@ref)) at the current per-particle dimension with a per-particle cotangent lift [`ParticlePSDLayer`](@ref), ``(q_i, p_i) \mapsto (\Phi q_i, \Phi p_i)`` with one ``\Phi`` shared by all particles, which raises the per-particle dimension from ``\tilde d`` up to ``d``; it ends with ambient pairwise shears at dimension ``d``. As in [`WeightTiedSymplecticAutoencoder`](@ref), one parameter tree `ps` is stored: [`decoder`](@ref)`(nn)` applies the layers in this order, and [`encoder`](@ref)`(nn)` applies their exact inverses ([`TransposeParticlePSDLayer`](@ref), i.e. ``\Phi^T``, and the sign-flipped shears) in reverse order with the same parameters.

A relabelling ``\sigma \in S_M`` acts on the ambient space by ``P_\sigma`` and on the latent space by the same relabelling ``R_\sigma`` of the latent particles. Every layer commutes with relabelling, so for all parameters `ps`
```math
\Psi^e \circ P_\sigma = R_\sigma \circ \Psi^e, \qquad \Psi^d \circ R_\sigma = P_\sigma \circ \Psi^d, \qquad \Psi^e \circ \Psi^d = \mathrm{id},
```
and the decoder is a symplectic embedding.

The states are a vector or a matrix whose first axis has length ``2Md`` (encoder input) or ``2M\tilde d`` (decoder input) in the flattened particle layout of [`SymmetricSymplecticAutoencoder`](@ref), or a `NamedTuple` `(q, p)` of particle arrays of size `(d, M)` or `(d, M, B)` or of vectors of length ``Md`` (``\tilde d`` in place of ``d`` for the latent state). A `(q, p)` input gives a `(q, p)` output in the same layout.

# Arguments

Besides the required integers `M >= 1`, `d >= 1` and `d_latent` (`1 <= d_latent <= d`), the following keyword arguments are supported:
- `n_blocks::Integer = 2`: Sets the spacing of the intermediate per-particle dimensions between `d_latent` and `d`, as `n_blocks` does for [`WeightTiedSymplecticAutoencoder`](@ref) (`n_blocks >= 1`, and `n_blocks <= d - d_latent + 1` when `d > d_latent`).
- `n_layers::Integer = 3`: Number of pairwise shears preceding each `ParticlePSDLayer`.
- `n_ambient_layers::Integer = 2`: Number of pairwise shears at dimension `d` following the final `ParticlePSDLayer`.
- `upscale::Integer = 3`: Per-particle upscaling factor of the pairwise shears.
- `activation = tanh`: Activation function of the pairwise shears.
- `init_q::Bool = true`: Whether the first pairwise shear in each block is a [`PairwiseSymplecticLayerQ`](@ref).
"""
struct EquivariantSymplecticAutoencoder{Init, AT} <: SymplecticCompression
    n_particles::Int
    particle_dim::Int
    latent_particle_dim::Int
    full_dim::Int
    reduced_dim::Int
    n_blocks::Int
    n_layers::Int
    n_ambient_layers::Int
    upscale::Int
    activation::AT
end

function EquivariantSymplecticAutoencoder(
        M::Integer, d::Integer, d_latent::Integer;
        n_blocks::Integer = 2,
        n_layers::Integer = 3,
        n_ambient_layers::Integer = 2,
        upscale::Integer = 3,
        activation = tanh,
        init_q::Bool = true)
    (M >= 1 && d >= 1 && 1 <= d_latent <= d) ||
        throw(ArgumentError(
            "EquivariantSymplecticAutoencoder requires M >= 1, d >= 1 and 1 <= d_latent <= d, got M = $M, d = $d, d_latent = $d_latent."))
    n_blocks >= 1 ||
        throw(ArgumentError("n_blocks ($n_blocks) must be at least 1."))
    (d == d_latent || n_blocks <= d - d_latent + 1) ||
        throw(ArgumentError(
            "n_blocks ($n_blocks) must be at most $(d - d_latent + 1) for d ($d) and d_latent ($d_latent)."))
    (n_layers >= 0 && n_ambient_layers >= 0 && upscale >= 1) ||
        throw(ArgumentError(
            "n_layers ($n_layers) and n_ambient_layers ($n_ambient_layers) must be non-negative and upscale ($upscale) must be at least 1."))
    Mp, dp, dl = Int(M), Int(d), Int(d_latent)
    EquivariantSymplecticAutoencoder{init_q ? :InitQ : :InitP, typeof(activation)}(
        Mp, dp, dl, 2 * Mp * dp, 2 * Mp * dl, Int(n_blocks), Int(n_layers),
        Int(n_ambient_layers), Int(upscale), activation)
end

function Chain(::EquivariantSymplecticAutoencoder)
    throw(ArgumentError(
        "An EquivariantSymplecticAutoencoder has no Chain of untied layers: build it with NeuralNetwork(arch) and take encoder(nn) and decoder(nn)."))
end

# The per-particle dimensions from `d_latent` up to `d`, spaced as the dimensions of
# `WeightTiedSymplecticAutoencoder`.
function _particle_dimensions(arch::EquivariantSymplecticAutoencoder)
    compute_iterations_for_symplectic_system(
        2 * arch.particle_dim, 2 * arch.latent_particle_dim, arch.n_blocks) .÷ 2
end

function _pairwise_shears(
        arch::EquivariantSymplecticAutoencoder, dk::Int, n::Int, init_q::Bool)
    Tuple((isodd(k) == init_q) ?
          PairwiseSymplecticLayerQ(arch.n_particles, dk, arch.upscale, arch.activation) :
          PairwiseSymplecticLayerP(arch.n_particles, dk, arch.upscale, arch.activation)
    for k in 1:n)
end

function decoder_layers_from_iteration(
        arch::EquivariantSymplecticAutoencoder{Init}) where {Init}
    init_q = Init === :InitQ
    dims = _particle_dimensions(arch)
    layers = ()
    for i in 1:(length(dims) - 1)
        layers = (layers..., _pairwise_shears(arch, dims[i], arch.n_layers, init_q)...,
            ParticlePSDLayer(arch.n_particles, dims[i], dims[i + 1]))
    end
    (layers..., _pairwise_shears(arch, arch.particle_dim, arch.n_ambient_layers, init_q)...)
end

# A chain of particle layers from `2 * Mp * d_in` (`N`) to `2 * Mp * d_out` (`n`) that also takes
# and returns `(q, p)` particle arrays. It is the model of `encoder(nn)` and `decoder(nn)`.
struct EquivariantParticleChainLayer{N, n, Mp, LT <: Tuple} <: AbstractExplicitLayer{N, n}
    layers::LT
end

function parameterlength(l::EquivariantParticleChainLayer)
    parameterlength(Chain(l.layers...))
end

function (l::EquivariantParticleChainLayer)(
        z::AbstractVecOrMat, ps::Union{NamedTuple, NetworkParameters})
    Chain(l.layers...)(z, ps)
end

function (l::EquivariantParticleChainLayer{N, n, Mp})(
        qp::NamedTuple{(:q, :p)}, ps::Union{NamedTuple, NetworkParameters}) where {
        N, n, Mp}
    d_in = N ÷ (2 * Mp)
    d_out = n ÷ (2 * Mp)
    qp_flat = _flatten_particle_qp(qp, Mp, d_in)
    y = l(vcat(qp_flat.q, qp_flat.p), ps)
    y_qp = assign_q_and_p(y, n ÷ 2)
    ndims(qp.q) == 1 ? y_qp : _unflatten_particle_qp(y_qp, Mp, d_out)
end

struct EquivariantAutoencoderLayer{N, n, Mp, ET <: Tuple, DT <: Tuple} <:
       AbstractExplicitLayer{N, N}
    encoder_layers::ET
    decoder_layers::DT
end

function EquivariantAutoencoderLayer(arch::EquivariantSymplecticAutoencoder)
    dec_layers = decoder_layers_from_iteration(arch)
    enc_layers = reverse(map(inverse_layer, dec_layers))
    N, n, Mp = arch.full_dim, arch.reduced_dim, arch.n_particles
    EquivariantAutoencoderLayer{N, n, Mp, typeof(enc_layers), typeof(dec_layers)}(
        enc_layers, dec_layers)
end

function _encoder_layer(l::EquivariantAutoencoderLayer{N, n, Mp, ET}) where {N, n, Mp, ET}
    EquivariantParticleChainLayer{N, n, Mp, ET}(l.encoder_layers)
end

function _decoder_layer(l::EquivariantAutoencoderLayer{
        N, n, Mp, ET, DT}) where {N, n, Mp, ET, DT}
    EquivariantParticleChainLayer{n, N, Mp, DT}(l.decoder_layers)
end

function parameterlength(l::EquivariantAutoencoderLayer)
    parameterlength(Chain(l.decoder_layers...))
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        l::EquivariantAutoencoderLayer, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T}
    initialparameters(rng, initializer, Chain(l.decoder_layers...), backend, T; kwargs...)
end

function (l::EquivariantAutoencoderLayer)(
        z::AbstractVecOrMat, ps::Union{NamedTuple, NetworkParameters})
    _decoder_layer(l)(_encoder_layer(l)(z, _reverse_params(ps)), ps)
end

function (l::EquivariantAutoencoderLayer)(
        qp::NamedTuple{(:q, :p)}, ps::Union{NamedTuple, NetworkParameters})
    _decoder_layer(l)(_encoder_layer(l)(qp, _reverse_params(ps)), ps)
end

function NeuralNetwork(
        arch::EquivariantSymplecticAutoencoder, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(arch, EquivariantAutoencoderLayer(arch), backend, T; kwargs...)
end

function decoder_parameters(nn::NeuralNetwork{<:EquivariantSymplecticAutoencoder})
    params(nn)
end

function encoder_parameters(nn::NeuralNetwork{<:EquivariantSymplecticAutoencoder})
    _reverse_params(params(nn))
end

function encoder(nn::NeuralNetwork{<:EquivariantSymplecticAutoencoder})
    a = nn.architecture
    arch = NonLinearSymplecticEncoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.upscale, a.activation)
    NeuralNetwork(arch, _encoder_layer(nn.model), encoder_parameters(nn), networkbackend(nn))
end

function decoder(nn::NeuralNetwork{<:EquivariantSymplecticAutoencoder})
    a = nn.architecture
    arch = NonLinearSymplecticDecoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.n_ambient_layers,
        a.upscale, a.activation)
    NeuralNetwork(arch, _decoder_layer(nn.model), decoder_parameters(nn), networkbackend(nn))
end
