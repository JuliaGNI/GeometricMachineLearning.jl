"""
This is a PSD-like layer used for symplectic autoencoders. 
One layer has the following shape:

    |Φ 0|
A = |0 Φ|, where Φ is an element of the regular Stiefel manifold. 
"""

struct PSDLayer{M, N, Retraction} <: AbstractExplicitLayer{M, N}
end

default_retr = Geodesic()
function PSDLayer(M::Integer, N::Integer; Retraction=default_retr)
    @assert iseven(M)
    @assert iseven(N)
    PSDLayer{M, N, typeof(Retraction)}()
end

function parameterlength(::PSDLayer{M, N}) where {M, N}
    M÷2*(N÷2 - (M÷2+1)÷2)
end 

function initialparameters(backend::KernelAbstractions.Backend, T::Type, ::PSDLayer{M, N}, rng::AbstractRNG=Random.default_rng()) where {M, N}
    (weight =  N > M ? rand(backend, rng, StiefelManifold{T}, N÷2, M÷2) : rand(backend, rng, StiefelManifold{T}, M÷2, N÷2), )
end

function (::PSDLayer{M, N})(x::AbstractVecOrMat, ps::NamedTuple) where {M, N}
    dim = size(x, 1)
    @assert dim == M 

    q, p = assign_q_and_p(x, dim÷2)
    N > M ? vcat(ps.weight*q, ps.weight*p) : vcat(ps.weight'*q, ps.weight'*p)
end

function retraction(::PSDLayer{N, M, Geodesic}, B::NamedTuple{(:weight,),Tuple{AT}}) where {N, M, AT<:StiefelLieAlgHorMatrix}
    geodesic(B)
end

function retraction(::PSDLayer{N, M, Cayley}, B::NamedTuple{(:weight,),Tuple{AT}}) where {N, M, AT<:StiefelLieAlgHorMatrix}
    cayley(B)
end  