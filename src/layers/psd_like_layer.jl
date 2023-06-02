"""
This is a PSD-like layer used for symplectic autoencoders. 
One layer has the following shape:

    |Φ 0|
A = |0 Φ|, where Φ is an element of the regular Stiefel manifold. 
"""

struct PSDLayer{inverse, Retraction, F1} <: Lux.AbstractExplicitLayer
    N::Integer
    n::Integer
    init_weight::F1
end

default_retr = Geodesic()
function PSDLayer(N2::Integer, n2::Integer; inverse::Bool=false, Retraction=default_retr, init_weight=Lux.glorot_uniform)
    @assert iseven(N2)
    @assert iseven(n2)
    PSDLayer{inverse, typeof(Retraction), typeof(init_weight)}(N2÷2, n2÷2, init_weight)
end

Lux.parameterlength(d::PSDLayer) = Int(d.n*(d.N - (d.n+1)/2))

function Lux.initialparameters(rng::AbstractRNG, d::PSDLayer)
    (weight = d.init_weight(rng, StiefelManifold, d.N, d.n), )
end

function Lux.apply(d::PSDLayer{false}, x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    @assert length(x) == 2*d.n
    output₁ = zeros(d.N)
    output₂ = zeros(d.N)
    output₁ = ps.weight*x[1:d.n]
    output₂ = ps.weight*x[(d.n+1):(2*d.n)]
    vcat(output₁, output₂), st
end

function Lux.apply(d::PSDLayer{false}, A::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    @assert size(A, 1) == 2*d.n
    output₁ = ps.weight*A[1:d.n, :]
    output₂ = ps.weight*A[(d.n+1):(2*d.n), :]
    vcat(output₁, output₂), st
end

function Lux.apply(d::PSDLayer{true}, x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    @assert length(x) == 2*d.N
    output₁ = ps.weight'*x[1:d.N]
    output₂ = ps.weight'*x[(d.N+1):(2*d.N)]
    vcat(output₁, output₂), st
end

function Lux.apply(d::PSDLayer{true}, A::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    @assert size(A, 1) == 2*d.N
    output₁ = ps.weight'*A[1:d.N, :]
    output₂ = ps.weight'*A[(d.N+1):(2*d.N), :]
    vcat(output₁, output₂), st
end

(d::PSDLayer)(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple) = Lux.apply(d, x, ps, st)