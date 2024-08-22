using GeometricMachineLearning
using GeometricMachineLearning: global_section, Ω
using Quadmath: Float128

import LinearAlgebra

N = 10
n = 5

function symplectic_stiefel_manifold_tests(T, N, n)
    U = rand(SymplecticStiefelManifold{T}, 2*N, 2*n)
    check_val = check(U)
    print("ErrSympl",T,": ", check_val, "\n")
    #this is the version using symplectic Householder reflections
    S = global_section₂(U)
    global_section_error = LinearAlgebra.norm((inv(S)*U)[vcat(1:(N-n), (N+1):(2*N-n)), :])
    print("ErrGlobalSection",T,": ", global_section_error, "\n")
    
    J = PoissonTensor(eltype(U), N)
    Δ = rgrad(U, rand(eltype(U), 2*N, 2*n), J)
    print("error in vector space property", T, ": ", LinearAlgebra.norm(Δ'*J*U + U'*J*Δ), "\n")
    print("error lie algebra lift", T ,": ", LinearAlgebra.norm(Ω(U, Δ)*U - Δ), "\n")
end

@time symplectic_stiefel_manifold_tests(Float32, N, n)
print("\n")
@time symplectic_stiefel_manifold_tests(Float64, N, n)
print("\n")
@time symplectic_stiefel_manifold_tests(Float128, N, n)

