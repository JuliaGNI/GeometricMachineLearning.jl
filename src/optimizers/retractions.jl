"""
This implements some basic retractions.

TODO: test for Cayley vs Exp
"""
abstract type AbstractRetraction end 

struct Cayley <: AbstractRetraction end

struct Geodesic <: AbstractRetraction end

#function retraction(d::ManifoldLayer{Geodesic}, B::StiefelLieAlgHorMatrix)
#    Exp(B)
#end

function retraction(d::ManifoldLayer{Geodesic}, B::NamedTuple{(:weight, ), Tuple{AT}}) where AT <: StiefelLieAlgHorMatrix
    (weight = Exp(B.weight),)
end

function retraction(d::ManifoldLayer{Cayley}, B::NamedTuple{(:weight, ), Tuple{AT}}) where AT <: StiefelLieAlgHorMatrix
    (weight = Cayley(B.weight),)
end

function retraction(d::Lux.AbstractExplicitLayer, gx::NamedTuple)
    gx
end

#geodesics for the Euclidean metric -> probably can get rid of this, there is no obvious advantage in keeping this (maybe for testing)!
function Exp_euc(B::StiefelLieAlgHorMatrix, η::AbstractFloat)
    hcat(vcat(I(B.n), zeros(B.N-B.n,B.n)), vcat(B.A, B.B))*
        exp(η*hcat(vcat(B.A, I(n)), vcat(B.A*B.A-B.B'*B.B, B.A)))*StiefelProjection(B.N, B.n)*
        exp(-η*B.A)
end
function Exp_euc(Y::StiefelManifold, Δ::AbstractMatrix, η::AbstractFloat)
    A = Y'*Δ
    N, n = size(Y)
    hcat(Y, Δ)*(exp(η*hcat(vcat(A, I(n)), vcat(-Δ'*Δ, A)))*StiefelProjection(2*n, n))*exp(-η*A)
end

function Exp(B::StiefelLieAlgHorMatrix)
    N, n = B.N, B.n
    E = StiefelProjection(N, n)
    #expression from which matrix exponential and inverse have to be computed
    exponent = hcat(vcat(.5*B.A, .25*B.A^2 - B.B'*B.B), vcat(I(n), .5*B.A))
    StiefelManifold(
        E + hcat(vcat(.5*B.A, B.B), E)*𝔄(exponent)*vcat(I(n), .5*B.A)
    )
end

Exp(B::StiefelLieAlgHorMatrix, η::AbstractFloat) = Exp(η*B)

function Exp(Y::StiefelManifold, Δ::AbstractMatrix, η::AbstractFloat)
    HD, B = global_rep(Y, Δ)
    apply_λ(Y, HD,  Exp(B, η))
end

function Cayley(B::StiefelLieAlgHorMatrix)
    N, n = B.N, B.n
    E = StiefelProjection(N, n)
    exponent = I - .5*hcat(vcat(.5*B.A, .25*B.A^2 - B.B'*B.B), vcat(I(n), .5*B.A))
    StiefelManifold(
        (I + .5*B)*
        (
            E + hcat(vcat(.25*B.A, .5*B.B), vcat(0.5*I(n), zero(B.B)))*(vcat(I(n), 0.5*B.A)/exponent)
            )
    )
end