const sa_symmetric_default::Bool = true
const sa_activation_default::AbstractSoftmax = MatrixSoftmax()

@doc raw"""
    SymplecticAttention

Implements the symplectic attention layers. See [`LinearSymplecticAttention`](@ref).

# Keys

It stores the following key:
- `activation::``AbstractSoftmax`

# Constructors

See [`SymplecticAttentionQ`](@ref) and [`SymplecticAttentionP`](@ref).

# Implementation

`SymplecticAttention` is similar to [`MultiHeadAttention`](@ref) or [`VolumePreservingAttention`](@ref) in that it computes the scalar products of all vectors in a sequence of input vectors:
```math
C = Q^TAQ,
```
where ``Q`` is the ``q``-part of an input ``Z`` (see [`QPT`](@ref)). The matrix ``A`` is a weighting that is either symmetric or arbitrary (this can be adjusted with the key-word `symmetric::Bool`).

# Extended help

The symplectic attention mechanism is derived via computing the gradient of a separable Hamiltonian, as is also done in [`GSympNet`](@ref)s.
"""
struct SymplecticAttention{M, N, LayerType, Symmetric, AT <: AbstractSoftmax} <:
       AbstractExplicitLayer{M, N}
    activation::AT
end

"""
    SymplecticAttentionQ

A constant that is derived from [`SymplecticAttention`](@ref). This only changes the ``q``-part of the input.

# Constructor

```julia
SymplecticAttentionQ(M; symmetric::Bool, activation)
```

The default for the keywords are $(sa_symmetric_default) and $(sa_activation_default).

You may want to alter the activation function (either [`MatrixSoftmax`](@ref) or [`VectorSoftmax`](@ref)), but it is almost always better to leave the keyword `symmetric` at `true`.
"""
const SymplecticAttentionQ{M, N, Symmetric, AT} = SymplecticAttention{
    M, N, :Q, Symmetric, AT}

"""
    SymplecticAttentionP

A constant that is derived from [`SymplecticAttention`](@ref). This only changes the `p`-part of the input.

# Constructor

```julia
SymplecticAttentionP(M; symmetric::Bool, activation)
```

The default for the keywords are $(sa_symmetric_default) and $(sa_activation_default).

You may want to alter the activation function (either [`MatrixSoftmax`](@ref) or [`VectorSoftmax`](@ref)), but it is almost always better to leave the keyword `symmetric` at `true`.
"""
const SymplecticAttentionP{M, N, Symmetric, AT} = SymplecticAttention{
    M, N, :P, Symmetric, AT}

# The keyword defaults are the two constants at the top of this file, and not literals repeated
# here: the docstrings above interpolate the constants, so reading the default off the constant is
# what makes the documented default the one a caller gets. The keyword's value is the type
# parameter, so the choice needs no branch.
function SymplecticAttentionQ(M::Integer; symmetric::Bool = sa_symmetric_default,
        activation::AbstractSoftmax = sa_activation_default)
    @assert iseven(M) "Dimension must be even!"
    SymplecticAttention{M, M, :Q, symmetric, typeof(activation)}(activation)
end
function SymplecticAttentionP(M::Integer; symmetric::Bool = sa_symmetric_default,
        activation::AbstractSoftmax = sa_activation_default)
    @assert iseven(M) "Dimension must be even!"
    SymplecticAttention{M, M, :P, symmetric, typeof(activation)}(activation)
end

function parameterlength(::SymplecticAttention{M, M, LayerType, false}) where {M, LayerType}
    M2 = M ÷ 2
    M2 * M2
end

function parameterlength(::SymplecticAttention{M, M, LayerType, true}) where {M, LayerType}
    M2 = M ÷ 2
    (M2 + 1) * M2 ÷ 2
end

function initialparameters(rng::AbstractRNG,
        initializer::AbstractNeuralNetworks.Initializer,
        l::SymplecticAttention{M, M, LayerType, true},
        backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, LayerType, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(l))
    initializer(rng, S)
    (A = SymmetricMatrix(S, M ÷ 2),)
end

function initialparameters(rng::AbstractRNG,
        initializer::AbstractNeuralNetworks.Initializer,
        ::SymplecticAttention{M, M, LayerType, false},
        backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, LayerType, T}
    A = KernelAbstractions.allocate(backend, T, M ÷ 2, M ÷ 2)
    initializer(rng, A)
    (A = A,)
end

# The weighting `A` weights the scalar products of the columns of the half that this layer does not
# change, and the layer adds the change to the one it does.
#
# `A = ps.A` once, and then from the binding: the local binding is what keeps the `SymmetricMatrix`
# structure in the Zygote gradient of this layer. Read `ps.A` twice and the gradient leaf degrades
# to a plain `Matrix` -- `test/layers/symplectic_attention_network_parameters_gradient.jl` asserts
# it, and the last testset there covers this method. The binding costs no allocation; it is the
# number of accesses that decides the structure.
function (d::SymplecticAttention{M, M, C, true})(
        z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M, C}
    A = ps.A
    _shift(Val(C), z,
        x -> _mul(_mul(A, x), 2 * d.activation(_mul(_mul(_custom_transpose(x), A), x))))
end

# One access to `ps.A`, for the reason given at the symmetric method above.
function (d::SymplecticAttention{M, M, C, false})(
        z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M, C}
    A = ps.A
    _shift(Val(C), z,
        x -> begin
            σ = d.activation(_mul(_mul(_custom_transpose(x), A), x))
            _mul(_mul(A, x), _custom_transpose(σ)) + _mul(_mul(A', x), σ)
        end)
end

function (d::SymplecticAttention)(z::AbstractArray, ps::NamedTuple)
    apply_layer_to_nt_and_return_array(z, d, ps)
end

_custom_transpose(Z::AbstractMatrix) = Z'
_custom_transpose(z::AbstractArray{<:Number, 3}) = permutedims(z, (2, 1, 3))
