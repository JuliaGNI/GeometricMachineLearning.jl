using GeometricMachineLearning: global_section
using LinearAlgebra

function test_global_section(T, N, n)
    U = rand(SymplecticStiefelManifold{T}, 2*N, 2*n)
    Φ = global_section(U)
    λU = hcat(U[:,1:n], vcat(Φ[:,1:(N-n)], zeros(N, N-n)), U[:,(n+1):(2*n)], vcat(zeros(N, N-n), Φ[:,1:(N-n)]))
    J = PoissonTensor(N)
    err = norm(λU'*J*λU - J)
    print(err)
end

T = Float64
N = 20
n = 10
test_global_section(T, 20, 10)