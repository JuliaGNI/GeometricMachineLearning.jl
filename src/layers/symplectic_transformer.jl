@doc raw"""
Implements the symplectic transformer. Analogous to SympNet gradient layers it performs

```math 
\begin{pmatrix} Q \\ P \end{pmatrix} \mapsto \begin{pmatrix} Q + \nabla_PF \\ P \end{pmatrix},
```
where ``Q,\, P\in\mathbb{R}^{n\times{}T}`` and ``F(P) = \mathrm{Tr}(P \mathrm{softmax}(P^TAP) * P^T)``. 
"""
struct SymplecticTransformerLayer{M, N} <: AbstractExplicitLayer{M, N} end

const SymplecticTransformerLayerQ{M, N} = SymplecticTransformerLayer{M, N, :Q}
const SymplecticTransformerLayerP{M, N} = SymplecticTransformerLayer{M, N, :P}

function SymplecticTransformerLayerQ(M)
    SymplecticTransformerLayer{M, M, :Q}
end
function SymplecticTransformerLayerP(M)
    SymplecticTransformerLayer{M, M, :P}
end
 
parameterlength(::SymplecticTransformerLayer{M, N}) where {M, N} = (M÷2) ^ 2 

function initialparameters(backend::KernelAbstractions.Backend, T::Type, ::SymplecticTransformerLayer{M}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    A = KernelAbstractions.allocate(backend, T, M÷2, M÷2)
    initializer(rng, A)
    (A = A, )
end

(::SymplecticTransformerLayerQ)(z::NamedTuple, ps::NamedTuple) = (q = z.q + symplectic_transformer_potential_gradient(z.p, ps.A), p = z.p)
(::SymplecticTransformerLayerP)(z::NamedTuple, ps::NamedTuple) = (q = z.q, p = z.p + symplectic_transformer_potential_gradient(z.q, ps.A))
