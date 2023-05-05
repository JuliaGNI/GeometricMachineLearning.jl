#retraction is more general! function on layer!

struct StiefelLayer{F1, F2} <: ManifoldLayer
    N::Int
    n::Int
    retraction::F1
    init_weight::F2
end

function StiefelLayer(N::Int, n::Int; init_weight=Lux.glorot_uniform, retraction=Exp)
    return StiefelLayer{typeof(init_weight), typeof(retraction)}(N, n, initial_weight, retraction)
end

function Lux.initialparameters(rng::AbstractRNG, d::StiefelLayer)
    B = StiefelLieAlgHorMatrix(
        SkewSymMatrix(d.init_weight(rng, d.n*(d.n-1)÷2), d.n),
        d.init_weight(rng, d.N, d.n), 
        d.N,
        d.n
    )
    (weight = d.retraction(B), )
end

Lux.initialstates(rng::AbstractRNG, d::StiefelLayer) = NamedTuple()

Lux.parameterlength(d::StiefelLayer) = d.n*(d.n-1)÷2 + d.N*d.n

Lux.statelength(d::StiefelLayer) = 0

function (d::StiefelLayer)(x::AbstractVecOrMat, ps, st::NamedTuple)
    ps.weight*x, st
end

function gradient_step(d::StiefelLayer, ps::NamedTuple, dx::NamedTuple, η)