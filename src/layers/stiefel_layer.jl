"""
Defines a layer that performs simple multiplication with an element of the Stiefel manifold.
"""
struct StiefelLayer{M, N, retraction} <: ManifoldLayer{M, N, retraction} end

default_retr = Geodesic()
function StiefelLayer(n::Integer, N::Integer, Retraction::AbstractRetraction=default_retr)
    StiefelLayer{n, N, typeof(Retraction)}()
end

function AbstractNeuralNetworks.initialparameters(backend::KernelAbstractions.Backend, ::Type{T}, d::StiefelLayer{M,N}; rng::AbstractRNG=Random.default_rng()) where {M,N,T}
    (weight = N > M ? rand(backend, rng, StiefelManifold{T}, N, M) : rand(backend, rng, StiefelManifold{T}, M, N), )
end

function parameterlength(::StiefelLayer{M, N}) where {M, N}
    N > M ? M*(M-1)÷2 + (N-M)*M : N*(N-1)÷2 + (M-N)*N
end