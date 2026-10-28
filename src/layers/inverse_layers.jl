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

Make an instance of [`InverseGradientLayer`](@ref) that subtracts the gradient shift from the ``q`` coordinate (analytic inverse of [`GradientLayerQ`](@ref)).
"""
function InverseGradientLayerQ(M::Integer, upscaling_dimension::Integer, activation = identity)
    @assert iseven(M) && iseven(upscaling_dimension)
    InverseGradientLayer{Int(M), Int(M), typeof(activation), :Q}(
        Int(upscaling_dimension), activation)
end

@doc raw"""
    InverseGradientLayerP(n, upscaling_dimension, activation = identity)

Make an instance of [`InverseGradientLayer`](@ref) that subtracts the gradient shift from the ``p`` coordinate (analytic inverse of [`GradientLayerP`](@ref)).
"""
function InverseGradientLayerP(M::Integer, upscaling_dimension::Integer, activation = identity)
    @assert iseven(M) && iseven(upscaling_dimension)
    InverseGradientLayer{Int(M), Int(M), typeof(activation), :P}(
        Int(upscaling_dimension), activation)
end

function parameterlength(d::InverseGradientLayer)
    parameterlength(inverse_layer(d))
end

function initialparameters(
        rng::AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        d::InverseGradientLayer{M, N}, backend::Backend, ::Type{T};
        kwargs...) where {M, N, T}
    initialparameters(rng, init_weight, inverse_layer(d), backend, T; kwargs...)
end

_shift_inv(::Val{:Q}, z::NamedTuple, f) = (q = z.q - f(z.p), p = z.p)
_shift_inv(::Val{:P}, z::NamedTuple, f) = (q = z.q, p = z.p - f(z.q))

@inline function (d::InverseGradientLayer{M, M, TA, C})(
        x::NamedTuple, ps) where {M, TA, C}
    size(x.q, 1) == size(x.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    _shift_inv(Val(C), x,
        y -> _mul(ps.weight', ps.scale .* d.activation.(_mul(ps.weight, y) .+ ps.bias)))
end

@inline function (d::InverseGradientLayer{M})(x::AbstractArray, ps::NamedTuple) where {M}
    size(x, 1) == M || error("Dimension mismatch.")
    apply_layer_to_nt_and_return_array(x, d, ps)
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

const _ISAArb{M, L} = InverseSymplecticAttention{M, M, L, :arbitrary}
const _ISASym{M, L} = InverseSymplecticAttention{M, M, L, :symmetric}

function _attention_displacement(
        d::InverseSymplecticAttention{M, M, LayerType, :arbitrary},
        x::AbstractArray, ps::NamedTuple) where {M, LayerType}
    XAX = _mul(_mul(_custom_transpose(x), ps.A), x)
    σXAX = d.activation(XAX)
    _mul(_mul(ps.A, x), _custom_transpose(σXAX)) + _mul(_mul(ps.A', x), σXAX)
end

function _attention_displacement(
        d::InverseSymplecticAttention{M, M, LayerType, :symmetric},
        x::AbstractArray, ps::NamedTuple) where {M, LayerType}
    A = ps.A
    _mul(_mul(A, x), 2 * d.activation(_mul(_mul(_custom_transpose(x), A), x)))
end

function (d::InverseSymplecticAttention{M, M, LayerType})(
        z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {M, LayerType, AT}
    size(z.q, 1) == size(z.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    _shift_inv(Val(LayerType), z, x -> _attention_displacement(d, x, ps))
end

(d::_ISAArb)(z::_VQP, ps::NamedTuple) = _apply_attention_vector(d, z, ps)
(d::_ISASym)(z::_VQP, ps::NamedTuple) = _apply_attention_vector(d, z, ps)

function (d::InverseSymplecticAttention{M})(z::AbstractArray, ps::NamedTuple) where {M}
    size(z, 1) == M || error("Dimension mismatch.")
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
function inverse_layer(d::GradientLayer{M, M, TA, C}) where {M, TA, C}
    InverseGradientLayer{M, M, TA, C}(d.second_dim, d.activation)
end

function inverse_layer(d::InverseGradientLayer{M, M, TA, C}) where {M, TA, C}
    GradientLayer{M, M, TA, C}(d.second_dim, d.activation)
end

inverse_layer(::Type{GradientLayerQ}) = InverseGradientLayerQ
inverse_layer(::Type{InverseGradientLayerQ}) = GradientLayerQ
inverse_layer(::Type{GradientLayerP}) = InverseGradientLayerP
inverse_layer(::Type{InverseGradientLayerP}) = GradientLayerP
inverse_layer(::Type{GradientLayerQ{M, N}}) where {M, N} = InverseGradientLayerQ{M, N}
inverse_layer(::Type{InverseGradientLayerQ{M, N}}) where {M, N} = GradientLayerQ{M, N}
inverse_layer(::Type{GradientLayerP{M, N}}) where {M, N} = InverseGradientLayerP{M, N}
inverse_layer(::Type{InverseGradientLayerP{M, N}}) where {M, N} = GradientLayerP{M, N}
function inverse_layer(::Type{GradientLayerQ{M, N, TA}}) where {M, N, TA}
    InverseGradientLayerQ{
        M, N, TA}
end
function inverse_layer(::Type{InverseGradientLayerQ{M, N, TA}}) where {M, N, TA}
    GradientLayerQ{
        M, N, TA}
end
function inverse_layer(::Type{GradientLayerP{M, N, TA}}) where {M, N, TA}
    InverseGradientLayerP{
        M, N, TA}
end
function inverse_layer(::Type{InverseGradientLayerP{M, N, TA}}) where {M, N, TA}
    GradientLayerP{
        M, N, TA}
end

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
function inverse_layer(::Type{SymplecticAttentionQ{M, N}}) where {M, N}
    InverseSymplecticAttentionQ{
        M, N}
end
function inverse_layer(::Type{InverseSymplecticAttentionQ{M, N}}) where {M, N}
    SymplecticAttentionQ{
        M, N}
end
function inverse_layer(::Type{SymplecticAttentionP{M, N}}) where {M, N}
    InverseSymplecticAttentionP{
        M, N}
end
function inverse_layer(::Type{InverseSymplecticAttentionP{M, N}}) where {M, N}
    SymplecticAttentionP{
        M, N}
end

inverse_layer(::PSDLayer{M, N}) where {M, N} = TransposePSDLayer{N, M}()
inverse_layer(::TransposePSDLayer{M, N}) where {M, N} = PSDLayer{N, M}()
inverse_layer(::Type{PSDLayer{M, N}}) where {M, N} = TransposePSDLayer{N, M}
inverse_layer(::Type{TransposePSDLayer{M, N}}) where {M, N} = PSDLayer{N, M}
inverse_layer(::Type{PSDLayer}) = TransposePSDLayer
inverse_layer(::Type{TransposePSDLayer}) = PSDLayer
