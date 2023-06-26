#retraction is more general! function on layer!

struct StiefelLayer{F1, reverse, F2, F3} <: ManifoldLayer
    N::Integer
    n::Integer
    init_weight::F2
    activation::F3
end

default_retr = Geodesic()
function StiefelLayer(N::Integer, n::Integer, activation=identity; init_weight=Lux.glorot_uniform, Retraction::AbstractRetraction=default_retr, Transpose::Bool=false)
    StiefelLayer{typeof(Retraction), Transpose, typeof(init_weight), typeof(activation)}(N, n, init_weight, activation)
end

function Lux.initialparameters(rng::AbstractRNG, d::StiefelLayer)
    (weight = d.init_weight(rng, StiefelManifold, d.N, d.n), )
end

#Lux.initialstates(::AbstractRNG, ::StiefelLayer) = NamedTuple()

Lux.parameterlength(d::StiefelLayer) = d.n*(d.n-1)÷2 + (d.N-d.n)*d.n

Lux.statelength(d::StiefelLayer) = 0

function (d::StiefelLayer{Retraction, false})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple) where {Retraction}
    d.activation.(ps.weight*x), st
end

function (d::StiefelLayer{Retraction, true})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple) where {Retraction}
    d.activation.(p.weight'*x), st
end