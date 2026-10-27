@doc raw"""
    InverseGradientLayer{M, N, TA, C}(second_dim, activation)

Analytic symplectic inverse of [`GradientLayer`](@ref).

Given a [`GradientLayer`](@ref) that shifts one canonical coordinate by ``\nabla V`` evaluated at the conjugate coordinate, `InverseGradientLayer` subtracts the exact same gradient update using the identical parameter `NamedTuple` `ps`.
"""
struct InverseGradientLayer{M, N, TA, C} <: SympNetLayer{M, M}
    second_dim::Int
    activation::TA
end

const InverseGradientLayerQ{M, N, TA} = InverseGradientLayer{M, N, TA, :Q}
const InverseGradientLayerP{M, N, TA} = InverseGradientLayer{M, N, TA, :P}

@doc raw"""
    InverseGradientLayerQ(n, upscaling_dimension, activation = identity)
    InverseGradientLayerQ{M, N}(activation = identity)

Make an instance of [`InverseGradientLayer`](@ref) that subtracts the gradient shift from the ``q`` coordinate (analytic inverse of [`GradientLayerQ`](@ref)).
"""
function InverseGradientLayerQ(M::Integer, N::Integer, activation = identity)
    @assert iseven(M) && iseven(N)
    InverseGradientLayer{Int(M), Int(N), typeof(activation), :Q}(Int(N), activation)
end

@doc raw"""
    InverseGradientLayerP(n, upscaling_dimension, activation = identity)
    InverseGradientLayerP{M, N}(activation = identity)

Make an instance of [`InverseGradientLayer`](@ref) that subtracts the gradient shift from the ``p`` coordinate (analytic inverse of [`GradientLayerP`](@ref)).
"""
function InverseGradientLayerP(M::Integer, N::Integer, activation = identity)
    @assert iseven(M) && iseven(N)
    InverseGradientLayer{Int(M), Int(N), typeof(activation), :P}(Int(N), activation)
end

function InverseGradientLayerQ{M, N}(activation = identity) where {M, N}
    InverseGradientLayerQ(M, N, activation)
end

function InverseGradientLayerP{M, N}(activation = identity) where {M, N}
    InverseGradientLayerP(M, N, activation)
end

function GradientLayerQ{M, N}(activation = identity) where {M, N}
    GradientLayerQ(M, N, activation)
end

function GradientLayerP{M, N}(activation = identity) where {M, N}
    GradientLayerP(M, N, activation)
end

function parameterlength(d::InverseGradientLayer{M, N}) where {M, N}
    d.second_dim ÷ 2 * (M ÷ 2 + 2)
end

function initialparameters(
        rng::AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        d::InverseGradientLayer{M, N}, backend::Backend, ::Type{T};
        kwargs...) where {M, N, T}
    initialparameters(rng, init_weight, inverse_layer(d), backend, T; kwargs...)
end

_shift_inv(::Val{:Q}, z::NamedTuple, f) = (q = z.q - f(z.p), p = z.p)
_shift_inv(::Val{:P}, z::NamedTuple, f) = (q = z.q, p = z.p - f(z.q))

const _WeightBiasScaleParams = NamedTuple{(:weight, :bias, :scale)}
const _WeightBiasParams = NamedTuple{(:weight, :bias)}
const _KbaParams = NamedTuple{(:K, :b, :a)}
const _KbParams = NamedTuple{(:K, :b)}
const _AltGradPS = Union{_WeightBiasParams, _KbaParams, _KbParams}

function _gradient_layer_delta(σ, y::AbstractArray, ps::_WeightBiasScaleParams)
    _mul(ps.weight', ps.scale .* σ.(_mul(ps.weight, y) .+ ps.bias))
end

function _gradient_layer_delta(σ, y::AbstractArray, ps::_WeightBiasParams)
    _mul(ps.weight', σ.(_mul(ps.weight, y) .+ ps.bias))
end

function _gradient_layer_delta(σ, y::AbstractArray, ps::_KbaParams)
    _mul(ps.K', ps.a .* σ.(_mul(ps.K, y) .+ ps.b))
end

function _gradient_layer_delta(σ, y::AbstractArray, ps::_KbParams)
    _mul(ps.K', σ.(_mul(ps.K, y) .+ ps.b))
end

function (d::GradientLayer{M, M, TA, C})(x::NamedTuple, ps::_AltGradPS) where {M, TA, C}
    size(x.q, 1) == size(x.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    _shift(Val(C), x, y -> _gradient_layer_delta(d.activation, y, ps))
end

function (d::InverseGradientLayer{M, N, TA, C})(x::NamedTuple, ps) where {M, N, TA, C}
    size(x.q, 1) == size(x.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    _shift_inv(Val(C), x, y -> _gradient_layer_delta(d.activation, y, ps))
end

@doc raw"""
    InverseSymplecticAttention{M, N, LayerType, Symmetric, AT}(activation)

Analytic symplectic inverse of [`SymplecticAttention`](@ref).

Subtracts the attention gradient update using the identical parameter `NamedTuple` `ps = (A = ...,)`.
"""
struct InverseSymplecticAttention{M, N, LayerType, Symmetric, AT <: AbstractSoftmax} <:
       AbstractExplicitLayer{M, N}
    activation::AT
end

const InverseSymplecticAttentionQ{M, N, Symmetric, AT} = InverseSymplecticAttention{
    M, N, :Q, Symmetric, AT}
const InverseSymplecticAttentionP{M, N, Symmetric, AT} = InverseSymplecticAttention{
    M, N, :P, Symmetric, AT}

@doc raw"""
    InverseSymplecticAttentionQ(M; symmetric = sa_symmetric_default, activation = sa_activation_default)

Make an instance of [`InverseSymplecticAttention`](@ref) that subtracts the attention gradient shift from the ``q`` coordinate (analytic inverse of [`SymplecticAttentionQ`](@ref)).
"""
function InverseSymplecticAttentionQ(M::Integer; symmetric::Bool = sa_symmetric_default,
        activation::AbstractSoftmax = sa_activation_default)
    @assert iseven(M) "Dimension must be even!"
    S = symmetric ? :symmetric : :arbitrary
    InverseSymplecticAttention{Int(M), Int(M), :Q, S, typeof(activation)}(activation)
end

@doc raw"""
    InverseSymplecticAttentionP(M; symmetric = sa_symmetric_default, activation = sa_activation_default)

Make an instance of [`InverseSymplecticAttention`](@ref) that subtracts the attention gradient shift from the ``p`` coordinate (analytic inverse of [`SymplecticAttentionP`](@ref)).
"""
function InverseSymplecticAttentionP(M::Integer; symmetric::Bool = sa_symmetric_default,
        activation::AbstractSoftmax = sa_activation_default)
    @assert iseven(M) "Dimension must be even!"
    S = symmetric ? :symmetric : :arbitrary
    InverseSymplecticAttention{Int(M), Int(M), :P, S, typeof(activation)}(activation)
end

function InverseSymplecticAttentionQ(M::Integer, ::Integer; kwargs...)
    InverseSymplecticAttentionQ(M; kwargs...)
end

function InverseSymplecticAttentionP(M::Integer, ::Integer; kwargs...)
    InverseSymplecticAttentionP(M; kwargs...)
end

function SymplecticAttentionQ(M::Integer, ::Integer; kwargs...)
    SymplecticAttentionQ(M; kwargs...)
end

function SymplecticAttentionP(M::Integer, ::Integer; kwargs...)
    SymplecticAttentionP(M; kwargs...)
end

function parameterlength(l::InverseSymplecticAttention)
    parameterlength(inverse_layer(l))
end

function initialparameters(rng::AbstractRNG,
        initializer::AbstractNeuralNetworks.Initializer,
        l::InverseSymplecticAttention,
        backend::KernelAbstractions.Backend,
        ::Type{T}) where {T}
    initialparameters(rng, initializer, inverse_layer(l), backend, T)
end

const _VQP{AT <: AbstractVector} = NamedTuple{(:q, :p), Tuple{AT, AT}}
const _SAArb{M, L} = SymplecticAttention{M, M, L, :arbitrary}
const _SASym{M, L} = SymplecticAttention{M, M, L, :symmetric}
const _ISAArb{M, L} = InverseSymplecticAttention{M, M, L, :arbitrary}
const _ISASym{M, L} = InverseSymplecticAttention{M, M, L, :symmetric}

function _apply_attention_vector(d::AbstractExplicitLayer, z::_VQP, ps::NamedTuple)
    out = d((q = reshape(z.q, :, 1), p = reshape(z.p, :, 1)), ps)
    (q = vec(out.q), p = vec(out.p))
end

(d::_SAArb)(z::_VQP, ps::NamedTuple) = _apply_attention_vector(d, z, ps)
(d::_SASym)(z::_VQP, ps::NamedTuple) = _apply_attention_vector(d, z, ps)

function (d::InverseSymplecticAttention{M, M, LayerType, :arbitrary})(
        z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {M, LayerType, AT}
    _shift_inv(Val(LayerType), z,
        function (x)
            XAX = _mul(_mul(_custom_transpose(x), ps.A), x)
            σXAX = d.activation(XAX)
            _mul(_mul(ps.A, x), _custom_transpose(σXAX)) + _mul(_mul(ps.A', x), σXAX)
        end)
end

function (d::InverseSymplecticAttention{M, M, LayerType, :symmetric})(
        z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {M, LayerType, AT}
    A = ps.A
    _shift_inv(Val(LayerType), z,
        x -> _mul(_mul(A, x), 2 * d.activation(_mul(_mul(_custom_transpose(x), A), x))))
end

(d::_ISAArb)(z::_VQP, ps::NamedTuple) = _apply_attention_vector(d, z, ps)
(d::_ISASym)(z::_VQP, ps::NamedTuple) = _apply_attention_vector(d, z, ps)

function (d::InverseSymplecticAttention)(z::AbstractArray, ps::NamedTuple)
    apply_layer_to_nt_and_return_array(z, d, ps)
end

@doc raw"""
    inverse_layer(l)

Return the analytic symplectic inverse (or Stiefel transpose for [`PSDLayer`](@ref)) of a layer `l` or layer type, sharing the same parameter `NamedTuple` `ps` without duplicating or mutating weights:

- `GradientLayerQ` ``\leftrightarrow`` [`InverseGradientLayerQ`](@ref)
- `GradientLayerP` ``\leftrightarrow`` [`InverseGradientLayerP`](@ref)
- `SymplecticAttentionQ` ``\leftrightarrow`` [`InverseSymplecticAttentionQ`](@ref)
- `SymplecticAttentionP` ``\leftrightarrow`` [`InverseSymplecticAttentionP`](@ref)
- `PSDLayer{2n, 2N}` ``\leftrightarrow`` [`TransposePSDLayer`](@ref)`{2N, 2n}`
"""
function inverse_layer(d::GradientLayer{M, M, TA, :Q}) where {M, TA}
    InverseGradientLayerQ(M, d.second_dim, d.activation)
end

function inverse_layer(d::GradientLayer{M, M, TA, :P}) where {M, TA}
    InverseGradientLayerP(M, d.second_dim, d.activation)
end

function inverse_layer(d::InverseGradientLayer{M, N, TA, :Q}) where {M, N, TA}
    GradientLayerQ(M, d.second_dim, d.activation)
end

function inverse_layer(d::InverseGradientLayer{M, N, TA, :P}) where {M, N, TA}
    GradientLayerP(M, d.second_dim, d.activation)
end

inverse_layer(::Type{GradientLayerQ}) = InverseGradientLayerQ
inverse_layer(::Type{InverseGradientLayerQ}) = GradientLayerQ
inverse_layer(::Type{GradientLayerP}) = InverseGradientLayerP
inverse_layer(::Type{InverseGradientLayerP}) = GradientLayerP

function inverse_layer(l::SymplecticAttention{M, M, L, S, AT}) where {M, L, S, AT}
    InverseSymplecticAttention{M, M, L, S, AT}(l.activation)
end

function inverse_layer(l::InverseSymplecticAttention{M, M, L, S, AT}) where {M, L, S, AT}
    SymplecticAttention{M, M, L, S, AT}(l.activation)
end

inverse_layer(::Type{SymplecticAttentionQ}) = InverseSymplecticAttentionQ
inverse_layer(::Type{InverseSymplecticAttentionQ}) = SymplecticAttentionQ
inverse_layer(::Type{SymplecticAttentionP}) = InverseSymplecticAttentionP
inverse_layer(::Type{InverseSymplecticAttentionP}) = SymplecticAttentionP

inverse_layer(::PSDLayer{M, N}) where {M, N} = TransposePSDLayer{N, M}()
inverse_layer(::TransposePSDLayer{M, N}) where {M, N} = PSDLayer{N, M}()
inverse_layer(::Type{PSDLayer{M, N}}) where {M, N} = TransposePSDLayer{N, M}
inverse_layer(::Type{TransposePSDLayer{M, N}}) where {M, N} = PSDLayer{N, M}
inverse_layer(::Type{PSDLayer}) = TransposePSDLayer
inverse_layer(::Type{TransposePSDLayer}) = PSDLayer
