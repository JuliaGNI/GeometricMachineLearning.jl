using GeometricMachineLearning, Test 
import Random 

Random.seed!(1234)

@doc raw"""
This function tests scalar multiplication for various custom arrays, i.e. if \((A,\alpha) \mapsto \alpha{}A\) is performed in the correct way. 
"""
function scalar_multiplication_for_custom_arrays(n::Int, N::Int, T::Type)
    A = rand(T, n, n)
    α = rand(T)

    # SymmetricMatrix
    Aα_sym = SymmetricMatrix(α * A)
    Aα_sym2 = α * SymmetricMatrix(A)
    @test Aα_sym ≈ Aα_sym2
    @test typeof(Aα_sym) <: SymmetricMatrix{T}
    @test typeof(Aα_sym2) <: SymmetricMatrix{T}

    # SkewSymMatrix
    Aα_skew = SkewSymMatrix(α * A)
    Aα_skew2 = α * SkewSymMatrix(A)
    @test Aα_skew ≈ Aα_skew2 
    @test typeof(Aα_skew) <: SkewSymMatrix{T}
    @test typeof(Aα_skew2) <: SkewSymMatrix{T}

    C = rand(T, N, N)

    # StiefelLieAlgHorMatrix
    Cα_slahm = StiefelLieAlgHorMatrix(α * C, n)
    Cα_slahm2 = α * StiefelLieAlgHorMatrix(C, n)
    @test Cα_slahm ≈ Cα_slahm2
    @test typeof(Cα_slahm) <: StiefelLieAlgHorMatrix{T}
    @test typeof(Cα_slahm2) <: StiefelLieAlgHorMatrix{T}

    Cα_glahm = GrassmannLieAlgHorMatrix(α * C, n)
    Cα_glahm2 = α * GrassmannLieAlgHorMatrix(C, n)
    @test Cα_glahm ≈ Cα_glahm2
    @test typeof(Cα_glahm) <: GrassmannLieAlgHorMatrix{T}
    @test typeof(Cα_glahm2) <: GrassmannLieAlgHorMatrix{T}
end

scalar_multiplication_for_custom_arrays(5, 10, Float32)