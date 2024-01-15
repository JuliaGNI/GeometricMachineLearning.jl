using GeometricMachineLearning, Test 
import Random

Random.seed!(1234)

@doc raw"""
This function tests matrix multiplication for various custom arrays, i.e. if \((A,\alpha) \mapsto \alpha{}A\) is performed in the correct way. 
"""
function matrix_multiplication_tests_for_custom_arrays(n::Int, N::Int, T::Type)
    A = rand(T, n, n)
    B = rand(T, n, N)

    # SymmetricMatrix
    A_sym = SymmetricMatrix(A)
    @test A_sym * B ≈ Matrix{T}(A_sym) * B 
    @test B' * A_sym ≈ B' * Matrix{T}(A_sym)

    # SkewSymMatrix
    A_skew = SkewSymMatrix(A)
    @test A_skew * B ≈ Matrix{T}(A_skew) * B 
    @test B' * A_skew ≈ B' * Matrix{T}(A_skew)
end

matrix_multiplication_tests_for_custom_arrays(5, 10, Float32)