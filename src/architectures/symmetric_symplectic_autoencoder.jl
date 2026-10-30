@doc raw"""
    SymmetricSymplecticAutoencoder(M, d, reduced_dim; frontend = :pairwise, kwargs...)

Make an instance of a symplectic autoencoder with a particle frontend for `M` particles in ``d`` spatial dimensions (ambient dimension ``2Md``) and latent dimension `reduced_dim` (``2n``).

`SymmetricSymplecticAutoencoder` combines a particle frontend (`frontend = :sort`, `:pairwise`, or `:slot`) with the weight-tied symplectic core [`WeightTiedSymplecticAutoencoder`](@ref) so that [`encoder`](@ref)`(nn)` and [`decoder`](@ref)`(nn)` share a single parameter tree.

With `frontend = :sort` or `frontend = :slot`, the latent state is invariant under a relabelling of particles with distinct coordinates along the sorting axis (`sort_axis` for `:sort`, coordinate 1 for `:slot`). With `frontend = :pairwise`, only the frontend is permutation-equivariant: the core is not, so the latent state is not invariant.
The sort of `:sort` and `:slot` is discontinuous in `p` where two particles cross along the sorting axis, so the Jacobian identity of the frontend holds only away from those crossings. `:sort` and `:slot` suit systems that keep their particle order, not a collisionless plasma, whose sorted snapshots are not low-rank.

The latent right-inversion ``\mathrm{encoder}(nn)(\mathrm{decoder}(nn)(\xi)) = \xi`` holds to machine precision for every ``\xi \in \mathbb{R}^{2n}`` with `frontend = :pairwise`. With `frontend = :sort` and `frontend = :slot`, it holds for ``\xi`` whose decoded state lies in the canonical chamber, where the sorting permutations on the decoded state are the identity; the decoder does not map into this chamber in general.

# Arguments

Besides `M::Integer`, `d::Integer`, and even integer `reduced_dim::Integer`:
- `frontend::Symbol = :pairwise`: Particle frontend (`:sort`, `:pairwise`, or `:slot`).
- `sort_axis::Integer = 1`: Coordinate axis (`1 <= sort_axis <= d`) used by [`CanonicalSortLayer`](@ref) when `frontend == :sort`.
- `n_pairwise_layers::Integer = 2`: Number of alternating [`PairwiseSymplecticLayerQ`](@ref) / [`PairwiseSymplecticLayerP`](@ref) shears when `frontend == :pairwise`.
- `pairwise_upscale::Integer = 3`: Per-particle upscaling factor in the pairwise symplectic layers.
- `n_slots::Integer = max(1, M - 1)`: Number of canonical slots (`1 <= n_slots <= M`) in [`CanonicalSlotLayer`](@ref) when `frontend == :slot`.
- `n_blocks::Integer = 2`, `n_layers::Integer = 3`, `n_ambient_layers::Integer = 2`, `sympnet_upscale::Integer = 5`, `activation = tanh`, `init_q::Bool = true`: Forwarded to the [`WeightTiedSymplecticAutoencoder`](@ref) core.
"""
struct SymmetricSymplecticAutoencoder{Frontend, Init, AT} <: SymplecticCompression
    n_particles::Int
    particle_dim::Int
    full_dim::Int
    reduced_dim::Int
    sort_axis::Int
    n_pairwise_layers::Int
    pairwise_upscale::Int
    n_slots::Int
    n_blocks::Int
    n_layers::Int
    n_ambient_layers::Int
    sympnet_upscale::Int
    activation::AT
end

function SymmetricSymplecticAutoencoder(
        M::Integer, d::Integer, reduced_dim::Integer;
        frontend::Symbol = :pairwise,
        sort_axis::Integer = 1,
        n_pairwise_layers::Integer = 2,
        pairwise_upscale::Integer = 3,
        n_slots::Integer = max(1, Int(M) - 1),
        n_blocks::Integer = 2,
        n_layers::Integer = 3,
        n_ambient_layers::Integer = 2,
        sympnet_upscale::Integer = 5,
        activation = tanh,
        init_q::Bool = true)
    (M >= 1 && d >= 1) ||
        throw(ArgumentError(
            "SymmetricSymplecticAutoencoder requires M >= 1 and d >= 1, got M = $M, d = $d."))
    frontend in (:sort, :pairwise, :slot) ||
        throw(ArgumentError(
            "frontend must be :sort, :pairwise, or :slot, got $(repr(frontend))."))
    1 <= sort_axis <= d ||
        throw(ArgumentError(
            "sort_axis ($sort_axis) must satisfy 1 <= sort_axis <= d ($d)."))
    (n_pairwise_layers >= 0 && pairwise_upscale >= 1) ||
        throw(ArgumentError(
            "n_pairwise_layers ($n_pairwise_layers) must be non-negative and pairwise_upscale ($pairwise_upscale) must be at least 1."))
    if frontend === :slot
        (M >= 2 && 1 <= n_slots <= M) ||
            throw(ArgumentError(
                "frontend = :slot requires M >= 2 and 1 <= n_slots <= M, got M = $M, n_slots = $n_slots."))
    end
    core_full_dim = frontend === :slot ? 2 * Int(n_slots) * Int(d) : 2 * Int(M) * Int(d)
    # Validate core arguments through WeightTiedSymplecticAutoencoder
    WeightTiedSymplecticAutoencoder(
        core_full_dim, reduced_dim;
        n_blocks = n_blocks, n_layers = n_layers,
        n_ambient_layers = n_ambient_layers,
        sympnet_upscale = sympnet_upscale,
        activation = activation, init_q = init_q)

    SymmetricSymplecticAutoencoder{frontend, init_q ? :InitQ : :InitP, typeof(activation)}(
        Int(M), Int(d), 2 * Int(M) * Int(d), Int(reduced_dim),
        Int(sort_axis), Int(n_pairwise_layers), Int(pairwise_upscale), Int(n_slots),
        Int(n_blocks), Int(n_layers), Int(n_ambient_layers), Int(sympnet_upscale),
        activation)
end

function Chain(::SymmetricSymplecticAutoencoder)
    throw(ArgumentError(
        "A SymmetricSymplecticAutoencoder has no Chain of untied layers: build it with NeuralNetwork(arch) and take encoder(nn) and decoder(nn)."))
end

function _core_architecture(arch::SymmetricSymplecticAutoencoder{
        Frontend, Init}) where {Frontend, Init}
    core_dim = Frontend === :slot ? 2 * arch.n_slots * arch.particle_dim : arch.full_dim
    WeightTiedSymplecticAutoencoder(
        core_dim, arch.reduced_dim;
        n_blocks = arch.n_blocks,
        n_layers = arch.n_layers,
        n_ambient_layers = arch.n_ambient_layers,
        sympnet_upscale = arch.sympnet_upscale,
        activation = arch.activation,
        init_q = Init === :InitQ)
end

function _frontend_decoder_layers(arch::SymmetricSymplecticAutoencoder{:sort})
    (InvertCanonicalSortLayer(arch.n_particles, arch.particle_dim),)
end

function _frontend_decoder_layers(arch::SymmetricSymplecticAutoencoder{:slot})
    (InverseCanonicalSlotLayer(arch.n_particles, arch.particle_dim, arch.n_slots),)
end

function _frontend_decoder_layers(arch::SymmetricSymplecticAutoencoder{
        :pairwise, Init}) where {Init}
    Mp = arch.n_particles
    dp = arch.particle_dim
    up = arch.pairwise_upscale
    act = arch.activation
    init_q = Init === :InitQ
    Tuple(
        (isodd(k) == init_q) ?
        InversePairwiseSymplecticLayerQ(Mp, dp, up, act) :
        InversePairwiseSymplecticLayerP(Mp, dp, up, act)
    for k in 1:(arch.n_pairwise_layers)
    )
end

function decoder_layers_from_iteration(arch::SymmetricSymplecticAutoencoder)
    core_arch = _core_architecture(arch)
    core_dec = decoder_layers_from_iteration(core_arch, compute_decoder_iterations(core_arch))
    fe_dec = _frontend_decoder_layers(arch)
    (core_dec..., fe_dec...)
end

function _flatten_particle_qp(qp::NamedTuple{(:q, :p)}, M::Int, d::Int)
    q, p = qp.q, qp.p
    N = M * d
    if ndims(q) == 2
        (size(q) == size(p) && size(q) == (d, M)) ||
            throw(DimensionMismatch(
                "Expected particle (q, p) of size ($d, $M), ($d, $M, B), or ($N,), got $(size(q)) and $(size(p))."))
        (q = vec(q), p = vec(p))
    elseif ndims(q) == 3
        (size(q) == size(p) && size(q, 1) == d && size(q, 2) == M) ||
            throw(DimensionMismatch(
                "Expected particle (q, p) of size ($d, $M), ($d, $M, B), or ($N,), got $(size(q)) and $(size(p))."))
        B = size(q, 3)
        (q = reshape(q, N, B), p = reshape(p, N, B))
    elseif ndims(q) == 1
        (length(q) == N && size(q) == size(p)) ||
            throw(DimensionMismatch(
                "Expected particle (q, p) of size ($d, $M), ($d, $M, B), or ($N,), got $(size(q)) and $(size(p))."))
        qp
    else
        throw(DimensionMismatch(
            "Expected particle (q, p) of size ($d, $M), ($d, $M, B), or ($N,), got $(size(q)) and $(size(p))."))
    end
end

function _unflatten_particle_qp(qp::NamedTuple{(:q, :p)}, M::Int, d::Int)
    q, p = qp.q, qp.p
    N = M * d
    if ndims(q) == 1
        length(q) == N ||
            throw(DimensionMismatch(
                "Expected flattened particle (q, p) of first dimension $N, got $(size(q)) and $(size(p))."))
        (q = reshape(q, d, M), p = reshape(p, d, M))
    elseif ndims(q) == 2
        size(q, 1) == N ||
            throw(DimensionMismatch(
                "Expected flattened particle (q, p) of first dimension $N, got $(size(q)) and $(size(p))."))
        B = size(q, 2)
        (q = reshape(q, d, M, B), p = reshape(p, d, M, B))
    else
        throw(DimensionMismatch(
            "Expected flattened particle (q, p) of first dimension $N, got $(size(q)) and $(size(p))."))
    end
end

struct SymmetricParticleEncoderLayer{M, d, Frontend, Axis, ET <: Tuple} <:
       AbstractExplicitLayer{M, M}
    encoder_layers::ET
end

struct SymmetricParticleDecoderLayer{M, d, Frontend, Axis, DT <: Tuple} <:
       AbstractExplicitLayer{M, M}
    decoder_layers::DT
end

struct SymmetricAutoencoderLayer{M, d, Frontend, Axis, ET <: Tuple, DT <: Tuple} <:
       AbstractExplicitLayer{M, M}
    encoder_layers::ET
    decoder_layers::DT
end

function SymmetricAutoencoderLayer(
        arch::SymmetricSymplecticAutoencoder{Frontend}) where {Frontend}
    dec_layers = decoder_layers_from_iteration(arch)
    enc_rest = reverse(map(inverse_layer, Base.front(dec_layers)))
    enc_first = Frontend === :sort ?
                CanonicalSortLayer(arch.n_particles, arch.particle_dim, arch.sort_axis) :
                inverse_layer(last(dec_layers))
    enc_layers = (enc_first, enc_rest...)
    M = arch.full_dim
    dp = arch.particle_dim
    ax = arch.sort_axis
    ET, DT = typeof(enc_layers), typeof(dec_layers)
    SymmetricAutoencoderLayer{M, dp, Frontend, ax, ET, DT}(enc_layers, dec_layers)
end

function parameterlength(l::SymmetricAutoencoderLayer)
    parameterlength(Chain(l.decoder_layers...))
end

function parameterlength(l::SymmetricParticleEncoderLayer)
    parameterlength(Chain(l.encoder_layers...))
end

function parameterlength(l::SymmetricParticleDecoderLayer)
    parameterlength(Chain(l.decoder_layers...))
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        l::SymmetricAutoencoderLayer, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T}
    initialparameters(rng, initializer, Chain(l.decoder_layers...), backend, T; kwargs...)
end

function (l::SymmetricParticleEncoderLayer{M, d, Frontend, Axis})(
        z::AbstractVecOrMat, ps::Union{NamedTuple, NetworkParameters}) where {
        M, d, Frontend, Axis}
    Chain(l.encoder_layers...)(z, ps)
end

function (l::SymmetricParticleEncoderLayer{M, d, Frontend, Axis})(
        qp::NamedTuple{(:q, :p)}, ps::Union{NamedTuple, NetworkParameters}) where {
        M, d, Frontend, Axis}
    Mp = M ÷ (2 * d)
    qp_flat = _flatten_particle_qp(qp, Mp, d)
    z = vcat(qp_flat.q, qp_flat.p)
    ξ = l(z, ps)
    assign_q_and_p(ξ, size(ξ, 1) ÷ 2)
end

function (l::SymmetricParticleDecoderLayer{M, d, Frontend, Axis})(
        ξ::AbstractVecOrMat, ps::Union{NamedTuple, NetworkParameters}) where {
        M, d, Frontend, Axis}
    Chain(l.decoder_layers...)(ξ, ps)
end

function (l::SymmetricParticleDecoderLayer{M, d, Frontend, Axis})(
        ξ::NamedTuple{(:q, :p)}, ps::Union{NamedTuple, NetworkParameters}) where {
        M, d, Frontend, Axis}
    Mp = M ÷ (2 * d)
    z = l(vcat(ξ.q, ξ.p), ps)
    qp_flat = assign_q_and_p(z, M ÷ 2)
    _unflatten_particle_qp(qp_flat, Mp, d)
end

function (l::SymmetricAutoencoderLayer{M, d, Frontend, Axis})(
        z::AbstractVecOrMat, ps::Union{NamedTuple, NetworkParameters}) where {
        M, d, Frontend, Axis}
    ps_enc = _reverse_params(ps)
    if Frontend === :sort
        Mp = M ÷ (2 * d)
        N = Mp * d
        sort_l = first(l.encoder_layers)
        inv_sort_l = last(l.decoder_layers)
        core_enc = Chain(Base.tail(l.encoder_layers)...)
        core_dec = Chain(Base.front(l.decoder_layers)...)
        ps_core_enc = NamedTuple{Base.tail(keys(ps_enc))}(Base.tail(values(ps_enc)))
        ps_core_dec = NamedTuple{Base.front(keys(ps))}(Base.front(values(ps)))
        z_mat = reshape(z, M, :)
        sorted = sort_l(
            (q = reshape(@view(z_mat[1:N, :]), d, Mp, :),
                p = reshape(@view(z_mat[(N + 1):M, :]), d, Mp, :)),
            NamedTuple())
        z_sorted = vcat(reshape(sorted.q, N, :), reshape(sorted.p, N, :))
        z_hat_sorted = core_dec(core_enc(z_sorted, ps_core_enc), ps_core_dec)
        unsorted = inv_sort_l(
            (q = reshape(@view(z_hat_sorted[1:N, :]), d, Mp, :),
                p = reshape(@view(z_hat_sorted[(N + 1):M, :]), d, Mp, :),
                perm = sorted.perm),
            NamedTuple())
        return reshape(vcat(reshape(unsorted.q, N, :), reshape(unsorted.p, N, :)), size(z))
    else
        ξ = Chain(l.encoder_layers...)(z, ps_enc)
        return Chain(l.decoder_layers...)(ξ, ps)
    end
end

function (l::SymmetricAutoencoderLayer{M, d, Frontend, Axis})(
        qp::NamedTuple{(:q, :p)}, ps::Union{NamedTuple, NetworkParameters}) where {
        M, d, Frontend, Axis}
    Mp = M ÷ (2 * d)
    qp_flat = _flatten_particle_qp(qp, Mp, d)
    z = vcat(qp_flat.q, qp_flat.p)
    z_rec = l(z, ps)
    qp_rec_flat = assign_q_and_p(z_rec, M ÷ 2)
    ndims(qp.q) == 1 ? qp_rec_flat : _unflatten_particle_qp(qp_rec_flat, Mp, d)
end

function NeuralNetwork(
        arch::SymmetricSymplecticAutoencoder, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(arch, SymmetricAutoencoderLayer(arch), backend, T; kwargs...)
end

function decoder_parameters(nn::NeuralNetwork{<:SymmetricSymplecticAutoencoder})
    params(nn)
end

function encoder_parameters(nn::NeuralNetwork{<:SymmetricSymplecticAutoencoder})
    _reverse_params(params(nn))
end

function encoder(
        nn::NeuralNetwork{<:SymmetricSymplecticAutoencoder{Frontend}}) where {Frontend}
    a = nn.architecture
    arch = NonLinearSymplecticEncoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.sympnet_upscale, a.activation)
    enc_layers = nn.model.encoder_layers
    model = SymmetricParticleEncoderLayer{
        a.full_dim, a.particle_dim, Frontend, a.sort_axis, typeof(enc_layers)}(enc_layers)
    NeuralNetwork(arch, model, encoder_parameters(nn), networkbackend(nn))
end

function decoder(
        nn::NeuralNetwork{<:SymmetricSymplecticAutoencoder{Frontend}}) where {Frontend}
    a = nn.architecture
    arch = NonLinearSymplecticDecoder(
        a.full_dim, a.reduced_dim, a.n_layers, a.n_blocks, a.n_ambient_layers,
        a.sympnet_upscale, a.activation)
    dec_layers = nn.model.decoder_layers
    model = SymmetricParticleDecoderLayer{
        a.full_dim, a.particle_dim, Frontend, a.sort_axis, typeof(dec_layers)}(dec_layers)
    NeuralNetwork(arch, model, decoder_parameters(nn), networkbackend(nn))
end
