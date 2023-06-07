#retraction is more general! function on layer!

struct StiefelLayer{F1, F2} <: ManifoldLayer
    N::Integer
    n::Integer
    init_weight::F2
end

default_retr = Geodesic()
function StiefelLayer(N::Integer, n::Integer; init_weight=Lux.glorot_uniform, Retraction::AbstractRetraction=default_retr)
    StiefelLayer{typeof(Retraction), typeof(init_weight)}(N, n, init_weight)
end

function Lux.initialparameters(rng::AbstractRNG, d::StiefelLayer)
    A = d.init_weight(rng, d.N, d.n)
    (weight = Lux.glorot_uniform(rng, d.N, d.n), )
end

#Lux.initialstates(::AbstractRNG, ::StiefelLayer) = NamedTuple()

Lux.parameterlength(d::StiefelLayer) = d.n*(d.n-1)÷2 + (d.N-d.n)*d.n

Lux.statelength(d::StiefelLayer) = 0

function (d::StiefelLayer)(x::AbstractVecOrMat, ps, st::NamedTuple)
    ps.weight*x, st
end