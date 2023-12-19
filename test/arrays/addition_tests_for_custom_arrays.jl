#=
This tests addition for all custom arrays. Note that these tests will also have to be performed on GPU!
=#

using LinearAlgebra
using Random
using Test
using GeometricMachineLearning

function test_addition_for_symmetric_matrix(n::Int, T::Type)
    A = rand(SymmetricMatrix{T}, n)
end