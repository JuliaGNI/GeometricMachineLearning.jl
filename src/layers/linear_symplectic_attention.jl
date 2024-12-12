@doc raw"""
    LinearSymplecticAttention

Implements the linear symplectic attention layers. Analogous to [`GradientLayer`](@ref) it performs mappings that only change the ``Q`` or the ``P`` part.

This layer preserves symplecticity in the *product-space sense*.

For more information see [`LinearSymplecticAttentionQ`](@ref) and [`LinearSymplecticAttentionP`](@ref).

# Implementation

The coefficients of a [`LinearSymplecticAttention`](@ref) layer is a [`SymmetricMatrix`](@ref):

```jldoctest
using GeometricMachineLearning

l = LinearSymplecticAttentionQ(3, 5)
ps = NeuralNetwork(Chain(l)).params.L1

typeof(ps.A) <: SymmetricMatrix

# output

true
```
"""
struct LinearSymplecticAttention{M, N, LayerType} <: AbstractExplicitLayer{M, N} 
    seq_length::Int
end

@doc raw"""
    LinearSymplecticAttentionQ(sys_dim, seq_length)

Make an instance of `LinearSymplecticAttentionQ` for a specific dimension and sequence length.

Performs: 

```math 
\begin{pmatrix} Q \\ P \end{pmatrix} \mapsto \begin{pmatrix} Q + \nabla_PF \\ P \end{pmatrix},
```
where ``Q,\, P\in\mathbb{R}^{n\times{}T}`` and ``F(P) = \frac{1}{2}\mathrm{Tr}(P A P^T)``.

The parameters of this layer are ``\bar{A} = \frac{1}{2}(A + A^T).``
"""
const LinearSymplecticAttentionQ{M, N} = LinearSymplecticAttention{M, N, :Q}

@doc raw"""
    LinearSymplecticAttentionP(sys_dim, seq_length)

Make an instance of `LinearSymplecticAttentionP` for a specific dimension and sequence length.

Performs: 

```math 
\begin{pmatrix} Q \\ P \end{pmatrix} \mapsto \begin{pmatrix} Q \\ P + \nabla_QF \end{pmatrix},
```
where ``Q,\, P\in\mathbb{R}^{n\times{}T}`` and ``F(Q) = \frac{1}{2}\mathrm{Tr}(Q A Q^T)``.

The parameters of this layer are ``\bar{A} = \frac{1}{2}(A + A^T).``
"""
const LinearSymplecticAttentionP{M, N} = LinearSymplecticAttention{M, N, :P}

function LinearSymplecticAttentionQ(M::Integer, seq_length::Integer)
    LinearSymplecticAttention{M, M, :Q}(seq_length)
end
function LinearSymplecticAttentionP(M::Integer, seq_length::Integer)
    LinearSymplecticAttention{M, M, :P}(seq_length)
end
 
parameterlength(l::LinearSymplecticAttention) = (l.seq_length + 1) * l.seq_length ÷ 2

function initialparameters(rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer, l::LinearSymplecticAttention, backend::KernelAbstractions.Backend, T::Type)
    S = KernelAbstractions.allocate(backend, T, parameterlength(l))
    initializer(rng, S)
    (A = SymmetricMatrix(S, l.seq_length), )
end

# Implement multiplication with symmetric matrix from the right!
function (::LinearSymplecticAttentionQ)(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where AT
    (q = z.q + _custom_mul(z.p, ps.A), p = z.p)
end

function (::LinearSymplecticAttentionP)(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where AT
    (q = z.q, p = z.p + _custom_mul(z.q, ps.A))
end

function (d::LinearSymplecticAttention)(z::AbstractArray, ps::NamedTuple)
    apply_layer_to_nt_and_return_array(z, d, ps)
end

_custom_mul(z::AbstractArray{T, 3}, A::AbstractMatrix{T}) where T = tensor_mat_mul(z, A)
_custom_mul(z::AbstractMatrix{T}, A::AbstractMatrix{T}) where T = z * A