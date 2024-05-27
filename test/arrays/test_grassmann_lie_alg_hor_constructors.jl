using GeometricMachineLearning
using Test
import Random 

Random.seed!(123)

function test_constructors(N::Integer, n::Integer; T::DataType=Float32)
    B = rand(T, N - n, n)
    B1 = GrassmannLieAlgHorMatrix(B, N, n)

    B2 = Matrix(B1) # note that this does not have any special structure

    B2 = GrassmannLieAlgHorMatrix(B2, n)
    
    E = StiefelProjection(B1)

    B3 = B1 * E

    B3 = GrassmannLieAlgHorMatrix(B3, n)

    @test B1 ≈ B2 ≈ B3
end

function test_lift(N::Integer, n::Integer; T::DataType=Float32)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    ΩΔ = GeometricMachineLearning.Ω(Y, Δ)
    λY = GlobalSection(Y) 

    λY_mat = Matrix(λY)

    Δ_lift1 =  λY_mat' * ΩΔ * λY_mat

    Δ_lift2 =  global_rep(λY, Δ)

    @test Δ_lift1 ≈ Δ_lift2
    @test ΩΔ * Y.A ≈ Δ
end

for T in (Float32, Float64)
    for N in (10, 20)
        for n in (3, 5)
            test_constructors(N, n; T = T)
            test_lift(N, n; T = T)
        end
    end
end