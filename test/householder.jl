"""
Some tests for the Householder reflections.
"""

using LinearAlgebra
using Test

include("../src/optimizers/householder.jl")
include("../src/optimizers/manifold_types.jl")

function is_upper_triangular(A::AbstractMatrix, ε)
    N, M = size(A)
    for i = 1:N
        for j = 1:(i-1) 
            @test abs(A[i,j]) < ε
        end
    end
end 

function qr_property_test(N::Int, ε=1e-12)
    A = randn(N,N)
    HD = HouseDecom(A)
    is_upper_triangular(HD'(A), ε)
end

function orthogonality_test(N::Int, n::Int, ε=1e-12)
    Y = StiefelManifold(N, n)
    A = rand(N, N-n)
    A = A - Y*Y'*A
    @test norm(HouseDecom(A)'(Y))/n < ε
end