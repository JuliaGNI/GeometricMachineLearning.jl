using GeometricMachineLearning
using LinearAlgebra
using Test 

function global_stiefel_section(N, n)
    Y = rand(StiefelManifold, N, n)
    λY = GlobalSection(Y)

    E = zeros(N, n)
    for i = 1:n
        E[i, i] = 1.
    end
    E = StiefelManifold(E)
    Y2 = apply_section(λY, E)
    @test typeof(Y2) <: StiefelManifold
    @test isapprox(Y2, Y)
end

function global_tangent_space_rep(N, n)
    Y = rand(StiefelManifold, N, n)
    λY = GlobalSection(Y)

    Δ = rgrad(Y, rand(N, n))
    B = global_rep(λY, Δ)
    BE = B*StiefelProjection(N, n)
    # abuse of notation
    Δ2 = apply_section(λY, StiefelManifold(BE))
    @test isapprox(Δ2, Δ)
end

N_max = 10
for N = 2:N_max
    for n = 2:N
        global_stiefel_section(N, n)
        global_tangent_space_rep(N, n)
    end
end