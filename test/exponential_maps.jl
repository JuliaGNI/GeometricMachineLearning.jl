using LinearAlgebra
using Test

include("../src/optimizers/auxiliary.jl")

#this function is only meant for testing purposes, to compare to standard exp
function exp_test(A::Matrix{T}) where T
    n, m = size(A)
    @assert n == m
    B = one(A)
    C = one(A)
    i = 1
    B_temp = zero(A)
    while norm(B) > eps(T)
        rmul!(B_temp, B, A)
        B .= B_temp
        rmul!(B, inv(i))
        C .+= B
        i += 1 
    end
    #print("Number of iterations is: ", i, "\n")
    C
end


N_vec = 2 .^ collect(7:12)
n_vec = 2 .^ collect(1:6)
ε = 1e-10
η = .1

for (N, n) ∈ zip(N_vec, n_vec)
    print("N = "*string(N)*", n = "*string(n)*"\n")
    Y = StiefelManifold(N, n)
    Δ = SkewSymMatrix(N)*Y
    @printf "Standard exponential:                                "
    @time sol₁ = exponential_retraction₁(Y, Δ, η)
    @printf "Exponential with householder (expected to be slower):"
    @time sol₂ = exponential_retraction₂(Y, Δ, η)
    @printf "Custom implementation (also gives householder):      "
    @time sol₃ = Exp(Y, Δ, η)
    @test norm(sol₁ - sol₂) < ε
    @test norm(sol₁ - sol₃) < ε
    print("\n")
end