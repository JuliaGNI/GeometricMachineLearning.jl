#retraction is more general! function on layer!

struct GrassmannLayer{N, M, reverse, retraction} <: ManifoldLayer{N, M, reverse, retraction} end

default_retr = Geodesic()
function GrassmannLayer(N::Integer, n::Integer, Transpose::Bool=false, Retraction::AbstractRetraction=default_retr)
    GrassmannLayer{N, n, Transpose, typeof(Retraction)}()
end

function AbstractNeuralNetworks.initialparameters(backend::KernelAbstractions.Backend, ::Type{T}, d::GrassmannLayer{N,M}; rng::AbstractRNG=Random.default_rng()) where {M,N,T}
    (weight = rand(backend, rng, GrassmannManifold{T}, N, M), )
end

#Lux.parameterlength(d::GrassmannLayer) = (d.N-d.n)*d.n