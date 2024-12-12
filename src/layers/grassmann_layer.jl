@doc raw"""
    GrassmannLayer(n, N)

Make an instance of `GrassmannLayer`.

This layer performs simple multiplication with an element of the Grassmann manifold, i.e.

```math
    \mathtt{GrassmannLayer}: x \mapsto Ax,
```
where ``A`` is a representation of an element in the Grassmann manifold, i.e. ``\mathcal{A} = \mathrm{span}(A)``.
"""
struct GrassmannLayer{M, N} <: ManifoldLayer{M, N} end

function GrassmannLayer(n::Integer, N::Integer)
    GrassmannLayer{n, N}()
end

function initialparameters(rng::AbstractRNG, init::AbstractNeuralNetworks.Initializer, ::GrassmannLayer{N,M}, backend::NeuralNetworkBackend, ::Type{T}) where {M,N,T}
    weight = N > M ? KernelAbstractions.allocate(backend, T, N, M) : KernelAbstractions.allocate(backend, T, M, N)
    init(rng, weight)
    (weight = GrassmannManifold(assign_columns(typeof(weight)(qr!(weight).Q), size(weight)...)), )
end

function parameterlength(::GrassmannLayer{M, N}) where {M, N}
    N > M ? (N - M)*M : (M - N):N
end