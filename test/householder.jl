"""
Some tests for the Householder reflections.
"""

using LinearAlgebra
using Test
using GeometricMachineLearning

#this is still included for now as it tests the Householder decomposition (same as LinearAlgebra.qr but we are using the LinearAlgebra routine at the moment)
include("../src/optimizers/householder.jl")

function is_upper_triangular(A::AbstractMatrix, ε=1e-12)
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

function qr_speed_test(N::Int, ε=1e-12)
    A = randn(N, N)
    print("Speed of LinearAlgebra.qr:\n")
    @time qr(A)
    print("Speed of GeometricMachineLearning.HouseDecom:\n") 
    @time HouseDecom(A) 
    print("\n")
end

N_max = 200
n_max = 50
num = 10
N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)
ε = 1e-11

for (N, n) ∈ zip(N_vec, n_vec)
    print("N = ", N, "\n")
    qr_property_test(N, ε)
    orthogonality_test(N, n, ε)
    qr_speed_test(N, ε)
end