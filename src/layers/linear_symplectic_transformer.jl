@doc raw"""
Implements the symplectic transformer. Analogous to SympNet gradient layers it performs

```math 
\begin{pmatrix} Q \\ P \end{pmatrix} \mapsto \begin{pmatrix} Q + \nabla_PF \\ P \end{pmatrix},
```
where ``Q,\, P\in\mathbb{R}^{n\times{}T}`` and ``F(P) = \mathrm{Tr}(P \mathrm{softmax}(P^TAP) * P^T)``. 
"""
struct LinearSymplecticTransformerLayer{M, N, LayerType} <: AbstractExplicitLayer{M, N} end

const LinearSymplecticTransformerLayerQ{M, N} = LinearSymplecticTransformerLayer{M, N, :Q}
const LinearSymplecticTransformerLayerP{M, N} = LinearSymplecticTransformerLayer{M, N, :P}

function LinearSymplecticTransformerLayerQ(M)
    LinearSymplecticTransformerLayer{M, M, :Q}()
end
function LinearSymplecticTransformerLayerP(M)
    LinearSymplecticTransformerLayer{M, M, :P}()
end
 
parameterlength(::LinearSymplecticTransformerLayer{M, N}) where {M, N} = (M÷2) ^ 2 

function initialparameters(backend::KernelAbstractions.Backend, T::Type, ::LinearSymplecticTransformerLayer{M}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    S = KernelAbstractions.allocate(backend, T, M, M)
    initializer(rng, S)
    (A = S, )
end

function (::LinearSymplecticTransformerLayerQ)(z::NamedTuple, ps::NamedTuple{(:q, :p), Tuple{AT, AT}}) where AT
    (q = z.q + tensor_mat_mul(z.p, ps.A + ps.A'), p = z.p)
end

function (::LinearSymplecticTransformerLayerP)(z::NamedTuple, ps::NamedTuple{(:q, :p), Tuple{AT, AT}}) where AT
    (q = z.q, p = z.p + tensor_mat_mul(z.q, ps.A + ps.A'))
end