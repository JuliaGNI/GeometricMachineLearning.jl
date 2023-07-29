"""
Defines a layer that performs simple multiplication with an element of the Grassmann manifold.
"""
struct GrassmannLayer{M, N, retraction} <: ManifoldLayer{M, N, retraction} end

default_retr = Geodesic()
function GrassmannLayer(n::Integer, N::Integer, Retraction::AbstractRetraction=default_retr)
    GrassmannLayer{n, N, typeof(Retraction)}()
end

function AbstractNeuralNetworks.initialparameters(backend::KernelAbstractions.Backend, ::Type{T}, d::GrassmannLayer{N,M}; rng::AbstractRNG=Random.default_rng()) where {M,N,T}
    (weight = N > M ? rand(backend, rng, GrassmannManifold{T}, N, M) : rand(backend, rng, GrassmannManifold{T}, M, N), )
end

function parameterlength(::GrassmannLayer{M, N}) where {M, N}
    N > M ? (N - M)*M : (M - N):N
end