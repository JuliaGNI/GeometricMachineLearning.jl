using GeometricMachineLearning, Test
using LinearAlgebra: I
import Random

Random.seed!(1234)

@doc raw"""
This tests various constructor for custom arrays, e.g. if calling `SymmetricMatrix` on a matrix ``A`` does 
```math 
A \mapsto \frac{1}{2}(A + A^T).
```
"""
function test_constructors_for_custom_arrays(n::Int, N::Int, T::Type)
    A = rand(T, n, n)
    B = rand(T, N, N)

    # SymmetricMatrix 
    @test Matrix{T}(SymmetricMatrix(A)) ≈ T(.5) * (A + A')

    # SkewSymMatrix 
    @test Matrix{T}(SkewSymMatrix(A)) ≈ T(.5) * (A - A')

    # StiefelLieAlgHorMatrix 
    B_shor = StiefelLieAlgHorMatrix(SkewSymMatrix(B), n)
    B_shor2 = Matrix{T}(SkewSymMatrix(B))
    B_shor2[(n+1):N, (n+1):N] .= zero(T)
    @test Matrix{T}(B_shor) ≈ B_shor2

    # GrassmannLieAlgHorMatrix 
    B_ghor = GrassmannLieAlgHorMatrix(SkewSymMatrix(B), n)
    B_ghor2 = copy(B_shor2)
    B_ghor2[1:n, 1:n] .= zero(T)
    @test Matrix{T}(B_ghor) ≈ B_ghor2 

    # StiefelProjection
    E = StiefelProjection(T, N, n)
    @test Matrix{T}(E) ≈ vcat(I(n), zeros(T, (N-n), n))
end

test_constructors_for_custom_arrays(5, 10, Float32)

@test LowerTriangular([1, 2, 3, 4, 5, 6], 4) ==
    [0 0 0 0; 1 0 0 0; 2 3 0 0; 4 5 6 0]
@test UpperTriangular([1, 2, 3, 4, 5, 6], 4) ==
    [0 1 2 4; 0 0 3 5; 0 0 0 6; 0 0 0 0]
@test SymmetricMatrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4) ==
    [1 2 4 7; 2 3 5 8; 4 5 6 9; 7 8 9 10]
@test SkewSymMatrix([1, 2, 3, 4, 5, 6], 4) ==
    [0 -1 -2 -4; 1 0 -3 -5; 2 3 0 -6; 4 5 6 0]
