const SYMPLECTICATTENTION_SYMMETRIC_DEFAULT::Bool = true
const SYMPLECTICATTENTION_ACTIVATION_DEFAULT::AbstractSoftmax = MatrixSoftmax()

@doc raw"""
    SymplecticAttention

Implements the symplectic attention layers. See [`LinearSymplecticAttention`](@ref).

# Keys

It stores the following key:
- `activation::`[`AbstractSoftmax`](@ref)

# Constructors

See [`SymplecticAttentionQ`](@ref) and [`SymplecticAttentionP`](@ref).

# Implementation

`SymplecticAttention` is similar to [`MultiHeadAttention`](@ref) or [`VolumePreservingAttention`](@ref) in that it computes the scalar products of all vectors in a sequence of input vectors:
```math
C = Q^TAQ,
```
where ``Q`` is the ``q``-part of an input ``Z`` (see [`QPT`](@ref)). The matrix ``A`` is a weighting that can either be symmetric or skew-symmetric (this can be adjusted with the key-word `symmetric::Bool`).

# Extended help

The symplectic attention mechanism is derived via computing the gradient of a separable Hamiltonian, as is also done in [`GSympNet`](@ref)s.
"""
struct SymplecticAttention{M, N, LayerType, Symmetric, AT <: AbstractSoftmax} <: AbstractExplicitLayer{M, N} 
    activation::AT
end

"""
    SymplecticAttentionQ

A constant that is derived from [`SymplecticAttention`](@ref). This only changes the ``q``-part of the input.

# Constructor

```julia
SymplecticAttentionQ(M; symmetric::Bool, activation)
```

The default for the keywords are $(SYMPLECTICATTENTION_SYMMETRIC_DEFAULT) and $(SYMPLECTICATTENTION_ACTIVATION_DEFAULT).

You may want to alter the activation function (either [`MatrixSoftmax`](@ref) or [`VectorSoftmax`](@ref)), but its almost always better to set the keyword `symmetric` to `true`.
"""
const SymplecticAttentionQ{M, N, Symmetric, AT} = SymplecticAttention{M, N, :Q, Symmetric, AT}

"""
    SymplecticAttentionP

A constant that is derived from [`SymplecticAttention`](@ref). This only changes the `p`-part of the input.

# Constructor

```julia
SymplecticAttentionP(M; symmetric::Bool, activation)
```

The default for the keywords are $(SYMPLECTICATTENTION_SYMMETRIC_DEFAULT) and $(SYMPLECTICATTENTION_ACTIVATION_DEFAULT).

You may want to alter the activation function (either [`MatrixSoftmax`](@ref) or [`VectorSoftmax`](@ref)), but its almost always better to set the keyword `symmetric` to `true`.
"""
const SymplecticAttentionP{M, N, Symmetric, AT} = SymplecticAttention{M, N, :P, Symmetric, AT}

function SymplecticAttentionQ(M::Integer; symmetric = false, activation::AbstractSoftmax = MatrixSoftmax())
    @assert iseven(M) "Dimension must be even!"
    AT = typeof(activation)
    symmetric == false ? SymplecticAttention{M, M, :Q, :arbitrary, AT}(activation) : SymplecticAttention{M, M, :Q, :symmetric, AT}(activation)
end
function SymplecticAttentionP(M::Integer; symmetric = false, activation::AbstractSoftmax = MatrixSoftmax())
    @assert iseven(M) "Dimension must be even!"
    AT = typeof(activation)
    symmetric == false ? SymplecticAttention{M, M, :P, :arbitrary, AT}(activation) : SymplecticAttention{M, M, :P, :symmetric, AT}(activation)
end
 
function parameterlength(::SymplecticAttention{M, M, LayerType, :arbitrary})::Integer where {M, LayerType} 
    M2 = M ÷ 2
    M2 * M2
end

function parameterlength(::SymplecticAttention{M, M, LayerType, :symmetric})::Integer where {M, LayerType}
    M2 = M ÷ 2
    (M2 + 1) * M2 ÷ 2
end

function initialparameters( rng::AbstractRNG, 
                            initializer::AbstractNeuralNetworks.Initializer, 
                            l::SymplecticAttention{M, M, LayerType, :symmetric}, 
                            backend::KernelAbstractions.Backend, 
                            T::Type) where {M, LayerType}
    S = KernelAbstractions.allocate(backend, T, parameterlength(l))
    initializer(rng, S)
    (A = SymmetricMatrix(S, M ÷ 2), )
end

function initialparameters( rng::AbstractRNG, 
                            initializer::AbstractNeuralNetworks.Initializer, 
                            ::SymplecticAttention{M, M, LayerType, :arbitrary}, 
                            backend::KernelAbstractions.Backend, 
                            T::Type) where {M, LayerType}
    A = KernelAbstractions.allocate(backend, T, M ÷ 2, M ÷ 2)
    initializer(rng, A)
    (A = A, )
end

function (d::SymplecticAttentionQ{M, M, :arbitrary})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    PAP = _custom_mul(_custom_mul(_custom_transpose(z.p), ps.A), z.p)
    σPAP = d.activation(PAP)
    (q = z.q + (_custom_mul(_custom_mul(ps.A, z.p), _custom_transpose(σPAP)) + _custom_mul(_custom_mul(ps.A', z.p), σPAP)), p = z.p)
end

function (d::SymplecticAttentionQ{M, M, :symmetric})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    A = ps.A # for some reason we have to allocate a local variable here. This should be further investigated.
    PAP = _custom_mul(_custom_mul(_custom_transpose(z.p), A), z.p)
    σPAP = d.activation(PAP)
    (q = z.q + _custom_mul(_custom_mul(A, z.p), 2 * σPAP), p = z.p)
end

function (d::SymplecticAttentionP{M, M, :arbitrary})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    QAQ = _custom_mul(_custom_mul(_custom_transpose(z.q), ps.A), z.q)
    σQAQ = d.activation(QAQ)
    (q = z.q, p = z.p + (_custom_mul(_custom_mul(ps.A, z.q), _custom_transpose(σQAQ)) + _custom_mul(_custom_mul(ps.A', z.q), σQAQ)))
end

function (d::SymplecticAttentionP{M, M, :symmetric})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    A = ps.A
    QAQ = _custom_mul(_custom_mul(_custom_transpose(z.q), A), z.q)
    σQAQ = d.activation(QAQ)
    (q = z.q, p = z.p + _custom_mul(_custom_mul(A, z.q), 2 * σQAQ))
end

function (d::SymplecticAttention)(z::AbstractArray, ps::NamedTuple)
    apply_layer_to_nt_and_return_array(z, d, ps)
end

_custom_mul(A::AbstractMatrix{T}, z::AbstractArray{T, 3}) where T = mat_tensor_mul(A, z)
_custom_mul(Z1::AbstractArray{T, 3}, Z2::AbstractArray{T, 3}) where T = tensor_tensor_mul(Z1, Z2)

_custom_transpose(Z::AbstractMatrix) = Z'
_custom_transpose(z::AbstractArray{T, 3}) where T = tensor_transpose(z)