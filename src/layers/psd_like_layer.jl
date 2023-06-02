"""
This is a PSD-like layer used for symplectic autoencoders. 
One layer has the following shape:

    |Φ 0|
A = |0 Φ|, where Φ is an element of the regular Stiefel manifold. 
"""

struct PSDLayer{inverse, Retraction, F1} <: Lux.AbstractExplicitLayer
    N::Integer
    n::Integer
    init_weight::F₁
end

default_retr = Geodesic()
function PSDLayer(N2::Integer, n2::Integer; inverse::Bool=false, Retraction=default_retr, init_weight=Lux.glorot_uniform)
    @assert iseven(N2)
    @assert iseven(n2)
    PSDLayer{typeof(Retraction), typeof(init_weight)}(N2÷2, n2÷2, init_weight)
end

Lux.parameterlength(d::PSDLayer) = Int(d.n(d.N - (d.n+1)/2))

function Lux.initialparameters(rng::AbstractRNG, d::PSDLayer)
    (weight = d.init_weight(rng, StiefelManifold, d.N, d.n), )
end

function Lux.apply(d::PSDLayer{false}, x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    @assert length(x) == 2*d.n
    output = zeros(2*d.N)
    output[1:N] = ps.weight*x[1:n]
    output[(N+1):(2*N)] = ps.weight*x[(n+1):(2*n)]
    output
end

function Lux.apply(d::PSDLayer{true}, x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    @assert length(x) == 2*d.N
    output = zeros(2*d.n)
    output[1:n] = ps.weight'*x[1:N]
    output[(n+1):(2*n)] = ps.weight'*x[(N+1):(2*N)]
    output
end