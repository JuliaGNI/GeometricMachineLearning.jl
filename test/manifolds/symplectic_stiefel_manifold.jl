using GeometricMachineLearning
using GeometricMachineLearning: global_section
using Quadmath: Float128

N = 50
n = 5

U = rand(SymplecticStiefelManifold, 2*N, 2*n)
check₁ = check(U)

U = rand(SymplecticStiefelManifold{Float32}, 2*N, 2*n)
check₂ = check(U)

U = rand(SymplecticStiefelManifold{Float128}, 2*N, 2*n)
check₃ = check(U)

print("ErrFloat64: ", check₁, "\n")
print("ErrFloat32: ", check₂, "\n")
print("ErrFloat128: ", check₃, "\n")

S = global_section(U);

LinearAlgebra.norm((inv(S)*U)[vcat(1:(N-n), (N+1):(2*N-n)), :])

J = SymplecticPotential(N÷2)
Δ = rgrad(U, rand(eltype(U), N, n), J)
Δ'*J*U + U'*J*Δ