"""
This implements some basic retractions.
"""

#geodesics for the Euclidean metric
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

Exp(Y::StiefelManifold, Δ::AbstractMatrix, η::AbstractFloat)

end