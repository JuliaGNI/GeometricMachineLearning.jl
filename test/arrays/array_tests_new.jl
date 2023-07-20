using LinearAlgebra
using Random
using Test

using BandedMatrices
using GeometricMachineLearning
"""
𝔤ʰ is the horizontal part of 𝔤, i.e. A ∈ 𝔤ʰ ⟺ AEE⁺ = A.

TODO: Add routine & test for symplectic conjugate
"""

#check if symmetric matrix works for 1×1 matrices 
W = rand(1,1)
S = SymmetricMatrix(W)
@test abs(W[1,1] - S[1,1]) < 1e-10

#check if built-in projection, matrix addition & subtraction works   
function sym_mat_add_sub_test(n)
    symmetrize(W) = .5*(W + W')
    W₁ = rand(n,n)
    S₁ = SymmetricMatrix(W₁)
    W₂ = rand(n,n)
    S₂ = SymmetricMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SymmetricMatrix
    @test typeof(S₄) <: SymmetricMatrix
    @test all(abs.(symmetrize(W₁ + W₂) .- S₃) .< 1e-10)
    @test all(abs.(symmetrize(W₁ - W₂) .- S₄) .< 1e-10)
end

function skew_mat_add_sub_test(n)
    anti_symmetrize(W) = .5*(W - W')
    W₁ = rand(n,n)
    S₁ = SkewSymMatrix(W₁)
    W₂ = rand(n,n)
    S₂ = SkewSymMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SkewSymMatrix
    @test typeof(S₄) <: SkewSymMatrix
    @test all(abs.(anti_symmetrize(W₁ + W₂) .- S₃) .< 1e-10)
    @test all(abs.(anti_symmetrize(W₁ - W₂) .- S₄) .< 1e-10)
end

#check if matrix is ∈ 𝔤 (check if the vector space projection works), addition & subtraction
function sympl_lie_alg_add_sub_test(n)
    J = SymplecticPotential(n)
    symplectisize(W) = .5*(W - J'*W'*J)
    W₁ = rand(2*n,2*n)
    S₁ = SymplecticLieAlgMatrix(W₁)
    W₂ = rand(2*n,2*n)
    S₂ = SymplecticLieAlgMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SymplecticLieAlgMatrix
    @test typeof(S₄) <: SymplecticLieAlgMatrix
    @test all(abs.(symplectisize(W₁ + W₂) .- S₃) .< 1e-10)
    @test all(abs.(symplectisize(W₁ - W₂) .- S₄) .< 1e-10)
end

#test Stiefel manifold projection test 
function stiefel_proj_test(N,n)
    In = I(n)
    E = StiefelProjection(N, n, Float64)
    @test all(abs.((E'*E) .- In) .< 1e-10)
end

#test symplectic projection (this is just the E matrix)
function sympl_proj_test(N, n)
    JN = SymplecticPotential(N)
    Jn = SymplecticPotential(n)
    E = SymplecticProjection(N, n, Float64)
    @test all(abs.((E'*JN*E) .- Jn) .< 1e-10)
end


function stiefel_lie_alg_add_sub_test(N, n)
    E = StiefelProjection(N, n)
    projection(W::SkewSymMatrix) = W - (I - E*E')*W*(I - E*E')
    W₁ = SkewSymMatrix(rand(N,N))
    S₁ = StiefelLieAlgHorMatrix(W₁,n)
    W₂ = SkewSymMatrix(rand(N,N))
    S₂ = StiefelLieAlgHorMatrix(W₂,n)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: StiefelLieAlgHorMatrix
    @test typeof(S₄) <: StiefelLieAlgHorMatrix
    @test all(abs.(projection(W₁ + W₂) .- S₃) .< 1e-10)
    @test all(abs.(projection(W₁ - W₂) .- S₄) .< 1e-10)
end

#check if matrix is ∈ 𝔤 (check if the vector space projection works), addition & subtraction
function sympl_lie_alg_add_sub_test(N, n)
    J = SymplecticPotential(n)
    E = SymplecticProjection(N, n)
    projection(W::SymplecticLieAlgMatrix) = W - (I - E*E')*W*(I - E*E')
    W₁ = SymplecticLieAlgMatrix(rand(2*N,2*N))
    S₁ = SymplecticLieAlgHorMatrix(W₁,n)
    W₂ = SymplecticLieAlgMatrix(rand(2*N,2*N))
    S₂ = SymplecticLieAlgHorMatrix(W₂,n)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SymplecticLieAlgHorMatrix
    @test typeof(S₄) <: SymplecticLieAlgHorMatrix
    @test all(abs.(projection(W₁ + W₂) .- S₃) .< 1e-10)
    @test all(abs.(projection(W₁ - W₂) .- S₄) .< 1e-10)
end


#TODO: tests for ADAM functions


#test everything for different n & N values
Random.seed!(42)

N_max = 20
n_max = 10
num = 100

N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)

for (N, n) ∈ zip(N_vec, n_vec)
    sym_mat_add_sub_test(N)
    skew_mat_add_sub_test(N)
    sympl_lie_alg_add_sub_test(N)
    stiefel_proj_test(N,n)
    sympl_proj_test(N,n)
    stiefel_lie_alg_add_sub_test(N,n)
    sympl_lie_alg_add_sub_test(N,n)
end
