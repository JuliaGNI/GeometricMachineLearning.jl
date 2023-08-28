"""
These may be useful for testing purposes, but are no longer used in src. 
"""


#I might actually not need this!
function Ω(U::SymplecticStiefelManifold{T}, Δ::AbstractMatrix{T}) where {T} 
    J_mat = SymplecticPotential(T, size(U,1)÷2)
    SymplecticLieAlgMatrix(
        Δ*inv(U'*U)*U' + J_mat*U*inv(U'U)*Δ'*(I + J_mat*U*inv(U'*U)*U'*J_mat)*J_mat
    )
end

Ω₁(Y::StiefelManifold, Δ::AbstractMatrix) = SkewSymMatrix(2*(I - .5*Y*Y')*Δ*Y') 
#TODO: perform calculations in-place, don't allocate so much!
function Ω(Y::StiefelManifold, Δ::AbstractMatrix)
    N = size(Y,1)
    B̃ = zeros(N, N)
    mul!(B̃, Δ, Y')
    B̂ = zero(B̃)
    mul!(B̂, Y, Y')
    rmul!(B̂, -.5)
    @views B̂ .+= one(B̂)
    B = zero(B̂)
    mul!(B, B̂, B̃)
    SkewSymMatrix(B)
end