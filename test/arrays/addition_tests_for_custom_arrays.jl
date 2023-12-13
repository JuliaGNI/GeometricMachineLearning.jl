using GeometricMachineLearning, Test 

@doc raw"""
This function tests addition for various custom arrays, i.e. if \(A + B\) is performed in the correct way. 
"""

function addition_tests_for_custom_arrays(n::Int, N::Int, T::Type)
    A = rand(T, n, n)
    B = rand(T, n, n)

    # SymmetricMatrix
    AB_sym = SymmetricMatrix(A + B)
    AB_sym2 = SymmetricMatrix(A) + SymmetricMatrix(B)
    @test AB_sym ≈ AB_sym2
    @test typeof(AB_sym) <: SymmetricMatrix{T}
    @test typeof(AB_sym2) <: SymmetricMatrix{T}

    # SkewSymMatrix
    AB_skew = SkewSymMatrix(A + B)
    AB_skew2 = SkewSymMatrix(A) + SkewSymMatrix(B)
    @test A_skew ≈ A_skew2 
    @test typeof(AB_skew) <: SkewSymMatrix{T}
    @test typeof(AB_skew2) <: SkewSymMatrix{T}

    C = rand(T, N, N)
    D = rand(T, N, N)

    # StiefelLieAlgHorMatrix
    CD_slahm = StiefelLieAlgHorMatrix(A + B, n)
    CD_slahm2 = StiefelLieAlgHorMatrix(A, n) + StiefelLieAlgHorMatrix(B, n)
    @test CD_slahm ≈ CD_slahm2
    @test typeof(CD_slahm) <: StiefelLieAlgHorMatrix{T}
    @test typeof(CD_slahm2) <: StiefelLieAlgHorMatrix{T}

end