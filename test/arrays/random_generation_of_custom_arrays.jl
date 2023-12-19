#=
This tests random generation of custom arrays. This will have to be expanded to GPU tests.
=#
using LinearAlgebra
using Random
using Test
using GeometricMachineLearning

function test_random_array_generation(n::Int, N::Int, T::Type)
    A_sym = rand(SymmetricMatrix{T}, n)
    @test typeof(A_sym) <: SymmetricMatrix{T}
    @test eltype(A_sym) == T

    A_skew = rand(SkewSymMatrix{T}, n)
    @test typeof(A_skew) <: SkewSymMatrix{T}
    @test eltype(A_skew) == T 

    A_stiefel_hor = rand(StiefelLieAlgHorMatrix{T}, N, n)
    @test typeof(A_stiefel_hor) <: StiefelLieAlgHorMatrix{T}
    @test eltype(A_stiefel_hor) == T

    A_grassmann_hor = rand(GrassmannLieAlgHorMatrix{T}, N, n)
    @test typeof(A_grassmann_hor) <: GrassmannLieAlgHorMatrix{T}
    @test eltype(A_grassmann_hor) == T
end

test_random_array_generation(5, 10, Float32)