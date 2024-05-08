@doc raw"""
Implements the symplectic transformer. Analogous to SympNet gradient layers it performs

```math 
\begin{pmatrix} Q \\ P \end{pmatrix} \mapsto \begin{pmatrix} Q + \nabla_PF \\ P \end{pmatrix},
```
where ``Q,\, P\in\mathbb{R}^{n\times{}T}`` and ``F(P) = \mathrm{Tr}(P \mathrm{softmax}(P^TAP) * P^T)``. 
"""
struct LinearSymplecticAttention{M, N, LayerType} <: AbstractExplicitLayer{M, N} end

const LinearSymplecticAttentionQ{M, N} = LinearSymplecticAttention{M, N, :Q}
const LinearSymplecticAttentionP{M, N} = LinearSymplecticAttention{M, N, :P}

function LinearSymplecticAttentionQ(M)
    LinearSymplecticAttention{M, M, :Q}()
end
function LinearSymplecticAttentionP(M)
    LinearSymplecticAttention{M, M, :P}()
end
 
parameterlength(::LinearSymplecticAttention{M, N}) where {M, N} = (M÷2) ^ 2 

function initialparameters(backend::KernelAbstractions.Backend, T::Type, ::LinearSymplecticAttention{M}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    S = KernelAbstractions.allocate(backend, T, M, M)
    initializer(rng, S)
    (A = S, )
end

function (::LinearSymplecticAttentionQ)(z::NamedTuple, ps::NamedTuple{(:q, :p), Tuple{AT, AT}}) where AT
    (q = z.q + tensor_mat_mul(z.p, ps.A + ps.A'), p = z.p)
end

function (::LinearSymplecticAttentionP)(z::NamedTuple, ps::NamedTuple{(:q, :p), Tuple{AT, AT}}) where AT
    (q = z.q, p = z.p + tensor_mat_mul(z.q, ps.A + ps.A'))
end