#retraction is more general! function on layer!

struct GrassmannLayer{F1, reverse, F2} <: ManifoldLayer
    N::Integer
    n::Integer
    init_weight::F2
end

default_retr = Geodesic()
function GrassmannLayer(N::Integer, n::Integer; init_weight=Lux.glorot_uniform, Retraction::AbstractRetraction=default_retr, Transpose::Bool=false)
    GrassmannLayer{typeof(Retraction), Transpose, typeof(init_weight)}(N, n, init_weight)
end

function Lux.initialparameters(rng::AbstractRNG, d::GrassmannLayer)
    (weight = d.init_weight(rng, GrassmannManifold, d.N, d.n), )
end

#Lux.initialstates(::AbstractRNG, ::GrassmannLayer) = NamedTuple()

Lux.parameterlength(d::GrassmannLayer) = (d.N-d.n)*d.n

Lux.statelength(d::GrassmannLayer) = 0

function (d::GrassmannLayer{Retraction, false})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple) where {Retraction}
    ps.weight*x, st
end

function (d::GrassmannLayer{Retraction, true})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple) where {Retraction}
    ps.weight'*x, st
end

#function Lux.apply(d::GrassmannLayer, x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple)
#    d(x, ps, st)
#end