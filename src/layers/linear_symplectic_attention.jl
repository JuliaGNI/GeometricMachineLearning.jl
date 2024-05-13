@doc raw"""
Implements the linear symplectic attention layers. Analogous to [`GradientLayer`](@ref) it performs mappings that only change the ``Q`` or the ``P`` part.
For more information see [`LinearSymplecticAttentionQ`](@ref) and [`LinearSymplecticAttentionP`](@ref).

### Constructor 

For the constructors simply call 

```julia
LinearSymplecticAttentionQ(sys_dim, seq_length)
``` or 

```julia
LinearSymplecticAttentionP(sys_dim, seq_length)
``` 
where `sys_dim` is the system dimension and `seq_length` is the sequence length.
"""
struct LinearSymplecticAttention{M, N, LayerType} <: AbstractExplicitLayer{M, N} 
    seq_length::Int
end

@doc raw"""
Performs: 

```math 
\begin{pmatrix} Q \\ P \end{pmatrix} \mapsto \begin{pmatrix} Q + \nabla_PF \\ P \end{pmatrix},
```
where ``Q,\, P\in\mathbb{R}^{n\times{}T}`` and ``F(P) = \frac{1}{2}\mathrm{Tr}(P A P^T)``. 
"""
const LinearSymplecticAttentionQ{M, N} = LinearSymplecticAttention{M, N, :Q}

@doc raw"""
Performs: 

```math 
\begin{pmatrix} Q \\ P \end{pmatrix} \mapsto \begin{pmatrix} Q + \nabla_PF \\ P \end{pmatrix},
```
where ``Q,\, P\in\mathbb{R}^{n\times{}T}`` and ``F(P) = \frac{1}{2}\mathrm{Tr}(P A P^T)``. 
"""
const LinearSymplecticAttentionP{M, N} = LinearSymplecticAttention{M, N, :P}

function LinearSymplecticAttentionQ(M::Integer, seq_length::Integer)
    LinearSymplecticAttention{M, M, :Q}(seq_length)
end
function LinearSymplecticAttentionP(M::Integer, seq_length::Integer)
    LinearSymplecticAttention{M, M, :P}(seq_length)
end
 
parameterlength(l::LinearSymplecticAttention) = (l.seq_length + 1) * l.seq_length ÷ 2

function initialparameters(l::LinearSymplecticAttention, backend::KernelAbstractions.Backend, T::Type; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform())
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