function _frontend_output_dimension(
        ::BilinearStiefelLayer{Nx, Nv, Mx, Mv}) where {Nx, Nv, Mx, Mv}
    2 * Mx * Mv
end

function _frontend_output_dimension(
        ::LowRankStiefelLayer{Nx, Nv, Mx, Mv}) where {Nx, Nv, Mx, Mv}
    2 * Mx * Mv
end

function _frontend_output_dimension(c::Chain)
    _frontend_output_dimension(last(c.layers))
end

function _frontend_output_dimension(l::AbstractExplicitLayer)
    output_dimension(l)
end

function _check_modality(frontends::NamedTuple{names}, modality::Symbol) where {names}
    modality in names || throw(ArgumentError(
        "Unknown modality :$modality. Available modalities: $names."))
    nothing
end

ChainRulesCore.@non_differentiable _check_modality(::Any, ::Any)

@doc raw"""
    MultiModalSymplecticAutoencoder(frontends::NamedTuple, core::SymplecticCompression)

Make an instance of a multi-modal symplectic autoencoder mapping a non-empty `NamedTuple` of modality frontends into a shared canonical latent space ``\xi = (\tilde{q}, \tilde{p}) \in \mathbb{R}^{2n}`` via `core`.

Each entry of `frontends` maps a modality symbol (such as `:particles`, `:spline`, `:grid`, or `:dlra`) to a symplectic or Poisson frontend layer whose canonical output dimension equals `core.full_dim = 2N`:
- [`CanonicalSortLayer`](@ref), [`CanonicalSlotLayer`](@ref), [`PairwiseSymplecticLayerQ`](@ref), [`PairwiseSymplecticLayerP`](@ref), or a `Chain` of particle layers with output dimension ``2N``;
- [`BilinearStiefelLayer`](@ref)`(Nx, Nv, Mx, Mv)` with ``2 M_x M_v = 2N`` (with or without `metric_factors` for spline vs grid modalities);
- [`LowRankStiefelLayer`](@ref)`(Nx, Nv, Mx, Mv; r_out = ...)` with ``2 M_x M_v = 2N``.

Given `nn = NeuralNetwork(arch, backend, T)`:
- [`encode`](@ref)`(nn, modality::Symbol, x)` applies `frontends[modality]` followed by `encoder(core)` to map modality input `x` into ``(\tilde{q}, \tilde{p}) \in \mathbb{R}^{2n}``;
- [`decode`](@ref)`(nn, modality::Symbol, ξ)` applies `decoder(core)` followed by `inverse_layer(frontends[modality])` to reconstruct in `modality`;
- [`translate`](@ref)`(nn, from::Symbol, to::Symbol, x)` computes `decode(nn, to, encode(nn, from, x))`.
"""
struct MultiModalSymplecticAutoencoder{
    FT <: NamedTuple,
    CT <: SymplecticCompression
} <: SymplecticCompression
    frontends::FT
    core::CT
    full_dim::Int
    reduced_dim::Int

    function MultiModalSymplecticAutoencoder(
            frontends::NamedTuple{names}, core::SymplecticCompression) where {names}
        isempty(names) && throw(
            ArgumentError(
            "MultiModalSymplecticAutoencoder requires a non-empty frontends NamedTuple."
        ),
        )
        N2 = core.full_dim
        n2 = core.reduced_dim
        for mod in names
            fe = getfield(frontends, mod)
            fe_dim = _frontend_output_dimension(fe)
            fe_dim == N2 || throw(
                ArgumentError(
                "Frontend :$mod has output dimension $fe_dim, expected core.full_dim = $N2."
            ),
            )
        end
        new{typeof(frontends), typeof(core)}(frontends, core, N2, n2)
    end
end

function AbstractNeuralNetworks.Chain(::MultiModalSymplecticAutoencoder)
    throw(
        ArgumentError(
        "A MultiModalSymplecticAutoencoder has no single Chain: build it with NeuralNetwork(arch) and use encode(nn, modality, x), decode(nn, modality, ξ), and translate(nn, from, to, x)."
    ),
    )
end

struct MultiModalSymplecticLayer{
    M,
    FT <: NamedTuple,
    IFT <: NamedTuple,
    CMT,
    CET,
    CDT
} <: AbstractExplicitLayer{M, M}
    frontends::FT
    inverse_frontends::IFT
    core_model::CMT
    core_encoder::CET
    core_decoder::CDT
    full_dim::Int
    reduced_dim::Int
end

function parameterlength(l::MultiModalSymplecticLayer)
    fe_len = sum(
        mod -> parameterlength(getfield(l.frontends, mod)), keys(l.frontends); init = 0
    )
    fe_len + parameterlength(l.core_model)
end

function _init_frontend_params(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        fe::AbstractExplicitLayer, backend::NeuralNetworkBackend, ::Type{T}) where {T}
    initialparameters(rng, initializer, fe, backend, T)
end

function _init_frontend_params(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        fe::Chain, backend::NeuralNetworkBackend, ::Type{T}) where {T}
    params(initialparameters(rng, initializer, fe, backend, T))
end

function NeuralNetwork(
        arch::MultiModalSymplecticAutoencoder,
        backend::NeuralNetworkBackend,
        ::Type{T};
        rng::AbstractRNG = Random.default_rng(),
        initializer::AbstractNeuralNetworks.Initializer = GlorotUniform(),
        kwargs...) where {T <: Number}
    names = keys(arch.frontends)
    fe_ps_vals = map(names) do mod
        fe = getfield(arch.frontends, mod)
        _init_frontend_params(rng, initializer, fe, backend, T)
    end
    fe_ps_nt = NamedTuple{names}(fe_ps_vals)
    inv_fe_vals = map(names) do mod
        inverse_layer(getfield(arch.frontends, mod))
    end
    inv_frontends = NamedTuple{names}(inv_fe_vals)

    core_nn = NeuralNetwork(arch.core, backend, T; rng = rng, kwargs...)
    core_enc_nn = encoder(core_nn)
    core_dec_nn = decoder(core_nn)

    M = arch.full_dim
    FT = typeof(arch.frontends)
    IFT = typeof(inv_frontends)
    CMT = typeof(core_nn.model)
    CET = typeof(core_enc_nn.model)
    CDT = typeof(core_dec_nn.model)
    layer = MultiModalSymplecticLayer{M, FT, IFT, CMT, CET, CDT}(
        arch.frontends,
        inv_frontends,
        core_nn.model,
        core_enc_nn.model,
        core_dec_nn.model,
        arch.full_dim,
        arch.reduced_dim
    )
    ps = NetworkParameters((frontends = fe_ps_nt, core = params(core_nn.params)))
    NeuralNetwork(arch, layer, ps, backend)
end

function encoder(nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder})
    core_nn = NeuralNetwork(
        nn.architecture.core, nn.model.core_model,
        NetworkParameters(params(nn).core), networkbackend(nn))
    encoder(core_nn)
end

function decoder(nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder})
    core_nn = NeuralNetwork(
        nn.architecture.core, nn.model.core_model,
        NetworkParameters(params(nn).core), networkbackend(nn))
    decoder(core_nn)
end

inverse_layer(c::Chain) = Chain(reverse(map(inverse_layer, c.layers))...)

_leaf_eltype(a::AbstractArray{T}) where {T <: Number} = T
_leaf_eltype(nt::NamedTuple) = _leaf_eltype(first(values(nt)))

_fe_output_to_core_vec(::Any, z_fe::AbstractVecOrMat) = z_fe

function _fe_output_to_core_vec(
        ::Union{BilinearStiefelLayer, LowRankStiefelLayer}, z_fe::NamedTuple)
    vcat(z_fe.q, z_fe.p)
end

function _fe_output_to_core_vec(::Any, z_fe::NamedTuple)
    if ndims(z_fe.q) <= 2
        vcat(vec(z_fe.q), vec(z_fe.p))
    else
        N = size(z_fe.q, 1) * size(z_fe.q, 2)
        B = size(z_fe.q, 3)
        vcat(reshape(z_fe.q, N, B), reshape(z_fe.p, N, B))
    end
end

function _narrow_core_vec(z_vec, x::NamedTuple, ::Type{T}) where {T}
    if ndims(first(values(x))) <= 2
        return z_vec::Vector{T}
    else
        return z_vec::Matrix{T}
    end
end

_narrow_core_vec(z_vec, ::AbstractVector, ::Type{T}) where {T} = z_vec::Vector{T}
function _narrow_core_vec(z_vec, ::AbstractArray{<:Any, 3}, ::Type{T}) where {T}
    z_vec::Matrix{T}
end
_narrow_core_vec(z_vec, ::Any, ::Type{T}) where {T} = z_vec

function _narrow_fe_vec(z_vec, ::Any, x::NamedTuple, ::Type{T}) where {T}
    _narrow_core_vec(z_vec, x, T)
end
_narrow_fe_vec(z_vec, ::Any, ::AbstractVector, ::Type{T}) where {T} = z_vec::Vector{T}
function _narrow_fe_vec(
        z_vec, ::BilinearStiefelLayer, ::AbstractMatrix, ::Type{T}
) where {T}
    z_vec::Vector{T}
end
_narrow_fe_vec(z_vec, ::Any, ::AbstractMatrix, ::Type{T}) where {T} = z_vec::Matrix{T}
function _narrow_fe_vec(z_vec, ::Any, ::AbstractArray{<:Any, 3}, ::Type{T}) where {T}
    z_vec::Matrix{T}
end
_narrow_fe_vec(z_vec, ::Any, ::Any, ::Type{T}) where {T} = z_vec

function _core_vec_to_fe_inv_input(
        ::Union{BilinearStiefelLayer, LowRankStiefelLayer},
        z_mid::AbstractVector, full_dim::Int)
    N = full_dim ÷ 2
    (q = z_mid[1:N], p = z_mid[(N + 1):full_dim])
end

function _core_vec_to_fe_inv_input(
        ::Union{BilinearStiefelLayer, LowRankStiefelLayer},
        z_mid::AbstractMatrix, full_dim::Int)
    N = full_dim ÷ 2
    (q = z_mid[1:N, :], p = z_mid[(N + 1):full_dim, :])
end

function _core_vec_to_fe_inv_input(
        ::Any, z_mid::AbstractVecOrMat, ::Int)
    z_mid
end

_apply_frontend(fe::AbstractExplicitLayer, x, ps_fe::NamedTuple) = fe(x, ps_fe)
_apply_frontend(fe::Chain, x::AbstractVecOrMat, ps_fe::NamedTuple) = fe(x, ps_fe)
function _apply_frontend(fe::Chain, x::NamedTuple{(:q, :p)}, ps_fe::NamedTuple)
    fe(_fe_output_to_core_vec(fe, x), ps_fe)
end

function _apply_inv_frontend(
        ::Chain, inv_fe::Chain, z_in, ps_fe::NamedTuple)
    inv_fe(z_in, _reverse_params(ps_fe))
end

function _apply_inv_frontend(
        ::AbstractExplicitLayer, inv_fe::AbstractExplicitLayer, z_in, ps_fe::NamedTuple)
    inv_fe(z_in, ps_fe)
end

function _core_encode_vec(
        ::Union{WeightTiedAutoencoderLayer, SymmetricAutoencoderLayer},
        core_encoder::Union{Chain, AbstractExplicitLayer},
        ps_core::NamedTuple, z_vec::AbstractVecOrMat)
    core_encoder(z_vec, _reverse_params(ps_core))
end

function _core_decode_vec(
        ::Union{WeightTiedAutoencoderLayer, SymmetricAutoencoderLayer},
        core_decoder::Union{Chain, AbstractExplicitLayer},
        ps_core::NamedTuple, ξ_vec::AbstractVecOrMat)
    core_decoder(ξ_vec, ps_core)
end

function _core_encode_vec(
        ::Any, core_encoder::Chain, ps_core::NamedTuple, z_vec::AbstractVecOrMat)
    names = keys(ps_core)
    M = length(core_encoder.layers)
    enc_names = names[1:M]
    ps_enc = NamedTuple{enc_names}(map(k -> getfield(ps_core, k), enc_names))
    core_encoder(z_vec, ps_enc)
end

function _core_decode_vec(
        ::Any, core_decoder::Chain, ps_core::NamedTuple, ξ_vec::AbstractVecOrMat)
    names = keys(ps_core)
    K = length(core_decoder.layers)
    M = length(names) - K
    dec_names = names[1:K]
    src_names = names[(M + 1):end]
    ps_dec = NamedTuple{dec_names}(map(k -> getfield(ps_core, k), src_names))
    core_decoder(ξ_vec, ps_dec)
end

_latent_to_vec(ξ::AbstractVecOrMat) = ξ
_latent_to_vec(ξ::NamedTuple) = vcat(ξ.q, ξ.p)

function _encode_active(
        fe, ps_fe::NamedTuple, core_model,
        core_encoder::Union{Chain, AbstractExplicitLayer}, ps_core::NamedTuple, x)
    z_fe = _apply_frontend(fe, x, ps_fe)
    T = promote_type(_leaf_eltype(x), _leaf_eltype(ps_core))
    z_vec = _narrow_fe_vec(_fe_output_to_core_vec(fe, z_fe), fe, x, T)
    ξ = _core_encode_vec(core_model, core_encoder, ps_core, z_vec)
    _narrow_fe_vec(ξ, fe, x, T)
end

function _decode_active(
        fe, inv_fe, ps_fe::NamedTuple, core_model,
        core_decoder::Union{Chain, AbstractExplicitLayer},
        ps_core::NamedTuple, full_dim::Int, ξ)
    ξ_vec = _latent_to_vec(ξ)
    z_mid = _core_decode_vec(core_model, core_decoder, ps_core, ξ_vec)
    z_in = _core_vec_to_fe_inv_input(fe, z_mid, full_dim)
    _apply_inv_frontend(fe, inv_fe, z_in, ps_fe)
end

_zero_param_tree(a::AbstractArray) = zero(parent(a))
_zero_param_tree(nt::NamedTuple) = map(_zero_param_tree, nt)

_resolve_param_grad(g::ChainRulesCore.AbstractZero, p_ref) = _zero_param_tree(p_ref)
_resolve_param_grad(::Nothing, p_ref) = _zero_param_tree(p_ref)
_resolve_param_grad(g::AbstractArray, ::AbstractArray) = g
function _resolve_param_grad(
        g::Union{ChainRulesCore.Tangent, NamedTuple},
        p_ref::NamedTuple{names}) where {names}
    NamedTuple{names}(
        map(
        k -> _resolve_param_grad(getproperty(g, k), getfield(p_ref, k)), names),
    )
end

@doc raw"""
    encode(nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, modality::Symbol, x)
    encode(model::MultiModalSymplecticLayer, ps, modality::Symbol, x)

Apply `frontends[modality]` followed by `encoder(core)` to map an input `x` in modality `modality` into the shared canonical latent space ``(\tilde{q}, \tilde{p}) \in \mathbb{R}^{2n}``.

Throws an `ArgumentError` naming `modality` and `keys(frontends)` if `modality` is not present in `frontends`.
"""
Base.@constprop :aggressive function encode(
        model::MultiModalSymplecticLayer, ps::NamedTuple, modality::Symbol, x)
    _check_modality(model.frontends, modality)
    fe = getfield(model.frontends, modality)
    ps_fe = getfield(ps.frontends, modality)
    T = promote_type(_leaf_eltype(x), _leaf_eltype(ps.core))
    ξ = _encode_active(fe, ps_fe, model.core_model, model.core_encoder, ps.core, x)
    _narrow_core_vec(ξ, x, T)
end

@noinline function _rrule_encode_active(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        fe, ps_fe::NamedTuple, core_model,
        core_encoder::Union{Chain, AbstractExplicitLayer}, ps_core::NamedTuple, x)
    ChainRulesCore.rrule_via_ad(
        config, _encode_active, fe, ps_fe, core_model, core_encoder, ps_core, x)
end

@noinline function _rrule_decode_active(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        fe, inv_fe, ps_fe::NamedTuple, core_model,
        core_decoder::Union{Chain, AbstractExplicitLayer},
        ps_core::NamedTuple, full_dim::Int, ξ)
    ChainRulesCore.rrule_via_ad(
        config, _decode_active, fe, inv_fe, ps_fe, core_model, core_decoder,
        ps_core, full_dim, ξ)
end

function ChainRulesCore.rrule(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        ::typeof(encode),
        model::MultiModalSymplecticLayer,
        ps::NamedTuple,
        modality::Symbol,
        x)
    _check_modality(model.frontends, modality)
    fe = getfield(model.frontends, modality)
    ps_fe = getfield(ps.frontends, modality)
    y, pb = _rrule_encode_active(
        config, fe, ps_fe, model.core_model, model.core_encoder, ps.core, x)
    function encode_pullback(Δy)
        _, _, d_ps_fe, _, _, d_ps_core, d_x = pb(ChainRulesCore.unthunk(Δy))
        fe_keys = keys(ps.frontends)
        d_frontends = NamedTuple{fe_keys}(
            map(
            k -> k === modality ?
                 _resolve_param_grad(
                ChainRulesCore.unthunk(d_ps_fe), getfield(ps.frontends, k)) :
                 _zero_param_tree(getfield(ps.frontends, k)),
            fe_keys),
        )
        d_core = _resolve_param_grad(ChainRulesCore.unthunk(d_ps_core), ps.core)
        d_ps = (frontends = d_frontends, core = d_core)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            d_ps,
            ChainRulesCore.NoTangent(),
            ChainRulesCore.unthunk(d_x)
        )
    end
    return y, encode_pullback
end

Base.@constprop :aggressive function encode(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, modality::Symbol, x)
    encode(model, params(ps), modality, x)
end

Base.@constprop :aggressive function encode(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, modality::Symbol, x)
    encode(nn.model, params(nn), modality, x)
end

@doc raw"""
    decode(nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, modality::Symbol, ξ)
    decode(model::MultiModalSymplecticLayer, ps, modality::Symbol, ξ)

Apply `decoder(core)` followed by `inverse_layer(frontends[modality])` to reconstruct a latent state ``\xi`` in modality `modality`.

Throws an `ArgumentError` naming `modality` and `keys(frontends)` if `modality` is not present in `frontends`.
"""
Base.@constprop :aggressive function decode(
        model::MultiModalSymplecticLayer, ps::NamedTuple, modality::Symbol, ξ)
    _check_modality(model.frontends, modality)
    fe = getfield(model.frontends, modality)
    inv_fe = getfield(model.inverse_frontends, modality)
    ps_fe = getfield(ps.frontends, modality)
    _decode_active(
        fe, inv_fe, ps_fe, model.core_model, model.core_decoder,
        ps.core, model.full_dim, ξ)
end

function ChainRulesCore.rrule(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        ::typeof(decode),
        model::MultiModalSymplecticLayer,
        ps::NamedTuple,
        modality::Symbol,
        ξ)
    _check_modality(model.frontends, modality)
    fe = getfield(model.frontends, modality)
    inv_fe = getfield(model.inverse_frontends, modality)
    ps_fe = getfield(ps.frontends, modality)
    y, pb = _rrule_decode_active(
        config, fe, inv_fe, ps_fe, model.core_model, model.core_decoder,
        ps.core, model.full_dim, ξ)
    function decode_pullback(Δy)
        _, _, _, d_ps_fe, _, _, d_ps_core, _, d_ξ = pb(ChainRulesCore.unthunk(Δy))
        fe_keys = keys(ps.frontends)
        d_frontends = NamedTuple{fe_keys}(
            map(
            k -> k === modality ?
                 _resolve_param_grad(
                ChainRulesCore.unthunk(d_ps_fe), getfield(ps.frontends, k)) :
                 _zero_param_tree(getfield(ps.frontends, k)),
            fe_keys),
        )
        d_core = _resolve_param_grad(ChainRulesCore.unthunk(d_ps_core), ps.core)
        d_ps = (frontends = d_frontends, core = d_core)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            d_ps,
            ChainRulesCore.NoTangent(),
            ChainRulesCore.unthunk(d_ξ)
        )
    end
    return y, decode_pullback
end

Base.@constprop :aggressive function decode(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, modality::Symbol, ξ)
    decode(model, params(ps), modality, ξ)
end

Base.@constprop :aggressive function decode(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, modality::Symbol, ξ)
    decode(nn.model, params(nn), modality, ξ)
end

@doc raw"""
    translate(nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, from::Symbol, to::Symbol, x)
    translate(model::MultiModalSymplecticLayer, ps, from::Symbol, to::Symbol, x)

Map an input `x` in modality `from` into modality `to` through the shared canonical latent space via `decode(nn, to, encode(nn, from, x))`.
"""
Base.@constprop :aggressive function translate(
        model::MultiModalSymplecticLayer, ps::NamedTuple, from::Symbol, to::Symbol, x)
    decode(model, ps, to, encode(model, ps, from, x))
end

Base.@constprop :aggressive function translate(
        model::MultiModalSymplecticLayer, ps::NetworkParameters,
        from::Symbol, to::Symbol, x)
    translate(model, params(ps), from, to, x)
end

Base.@constprop :aggressive function translate(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder},
        from::Symbol, to::Symbol, x)
    translate(nn.model, params(nn), from, to, x)
end

function (model::MultiModalSymplecticLayer)(x, ps::NamedTuple)
    mod = first(keys(model.frontends))
    decode(model, ps, mod, encode(model, ps, mod, x))
end
