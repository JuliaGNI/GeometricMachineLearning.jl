using Test 
using LinearAlgebra
using GeometricMachineLearning
using GeometricMachineLearning: Ω
import Random

Random.seed!(123)

N = 5
A = rand(N,N)
A_skew = SkewSymMatrix(A)

for i in 1:N 
    for j in 1:N 
        @test abs(.5*(A - A')[i,j] - A_skew[i,j]) < 1e-10
    end
end

n = 1
A_hor = StiefelLieAlgHorMatrix(A_skew, n)

for i in 1:n
    for j in 1:N 
        @test abs(A_hor[i,j] - A_skew[i,j]) < 1e-10
    end 
end

for i in (n+1):N 
    for j in 1:n 
        @test abs(A_hor[i,j] - A_skew[i,j]) < 1e-10
    end
    for j in (n+1):N 
        @test abs(A_hor[i,j]) < 1e-10
    end
end

function Ω_test(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{Float32}, 5, 3)
    Δ = rgrad(Y, rand(Float32, 5, 3))
    @test GeometricMachineLearning.Ω(Y, Δ) * Y.A ≈ Δ
end

function retraction_test(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = geodesic(Y, Δ / 1000)
    @test norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function metric_test(N, n, T)
    Y = rand(StiefelManifold{T}, N, n)
    Δ₁ = rgrad(Y, rand(T, N, n))
    Δ₂ = rgrad(Y, rand(T, N, n))
    @test T(.5) * tr(Ω(Y, Δ₁)' * Ω(Y, Δ₂)) ≈ metric(Y, Δ₁, Δ₂)
end

for N in (20, 10)
    for n in (5, 3)
        for T in (Float64, Float32)
            Ω_test(N, n, T)
            retraction_test(N, n, T)
            metric_test(N, n, T)
        end
    end
end