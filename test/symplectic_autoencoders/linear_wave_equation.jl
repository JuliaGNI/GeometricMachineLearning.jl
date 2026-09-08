using Test

include("../../scripts/symplectic_autoencoders/assemble_matrix.jl")

μ = 0.5
Δx = 0.1
Ñ = 2048

K = assemble_matrix(μ, Δx)

function H₁(q, μ, Δx, Ñ = length(q)-2)
    H_val = 0
    for i in 1:Ñ
        H_val += μ^2 / (2 * Δx) *
                 (q[i]*(q[i] - q[i - 1] - q[i + 1]) + (q[i - 1]^2 + q[i + 1]^2) / 2)
    end
    H_val
end

_mul(q₁::OffsetVector, A::OffsetMatrix, q₂::OffsetVector) = q₁.parent'*A.parent*q₂.parent

q_vec = OffsetArray(rand(Ñ+2), OffsetArrays.Origin(0))
@test isapprox(H₁(q_vec, μ, Δx), _mul(q_vec, K, q_vec))
