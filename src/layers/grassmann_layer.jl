#retraction is more general! function on layer!

struct GrassmannLayer{F1, F2} <: ManifoldLayer
    N::Integer
    n::Integer
    init_weight::F2
end

default_retr = Geodesic()
function GrassmannLayer(N::Integer, n::Integer; init_weight=Lux.glorot_uniform, Retraction::AbstractRetraction=default_retr)
    GrassmannLayer{typeof(Retraction), typeof(init_weight)}(N, n, init_weight)
end

function Lux.initialparameters(rng::AbstractRNG, d::GrassmannLayer)
    (weight = Lux.glorot_uniform(rng, GrassmannManifold, d.N, d.n), )
end

function Lux.initialparameters(::TrivialInitRNG, d::GrassmannLayer)
    (weight = zeros(StiefelLieAlgHorMatrix{Float32}, d.N, d.n), )
end

#Lux.initialstates(::AbstractRNG, ::GrassmannLayer) = NamedTuple()

Lux.parameterlength(d::GrassmannLayer) = (d.N-d.n)*d.n

Lux.statelength(d::GrassmannLayer) = 0

function (d::GrassmannLayer)(x::AbstractVecOrMat, ps, st::NamedTuple)
    ps.weight*x, st
end