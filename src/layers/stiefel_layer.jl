#retraction is more general! function on layer!

struct StiefelLayer{N, M, reverse, retraction} <: ManifoldLayer{N, M, reverse, retraction} end

default_retr = Geodesic()
function StiefelLayer(N::Integer, n::Integer, Transpose::Bool=false, Retraction::AbstractRetraction=default_retr)
    StiefelLayer{N, n, Transpose, typeof(Retraction)}()
end

function AbstractNeuralNetworks.initialparameters(backend::KernelAbstractions.Backend, ::Type{T}, d::StiefelLayer{N,M}; rng::AbstractRNG=Random.default_rng()) where {M,N,T}
    (weight = rand(backend, rng, StiefelManifold{T}, N, M), )
end

#Lux.parameterlength(d::StiefelLayer) = d.n*(d.n-1)÷2 + (d.N-d.n)*d.n