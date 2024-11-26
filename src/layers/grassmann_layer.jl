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

function AbstractNeuralNetworks.initialparameters(d::GrassmannLayer{N,M}, backend::KernelAbstractions.Backend, ::Type{T}; rng::AbstractRNG=Random.default_rng()) where {M,N,T}
    (weight = N > M ? rand(backend, rng, GrassmannManifold{T}, N, M) : rand(backend, rng, GrassmannManifold{T}, M, N), )
end

function parameterlength(::GrassmannLayer{M, N}) where {M, N}
    N > M ? (N - M)*M : (M - N):N
end