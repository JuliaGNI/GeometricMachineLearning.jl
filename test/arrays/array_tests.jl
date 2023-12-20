using LinearAlgebra
using Random
using Test
using GeometricMachineLearning

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

# this function tests if the matrix multiplication for the SkewSym Matrix is the same as the implied one.
function skew_mat_mul_test(n, T=Float64)
    S = rand(SkewSymMatrix{T}, n)
    A = rand(n, n)
    SA1 = S*A 
    SA2 = Matrix{T}(S)*A 
    @test isapprox(SA1, SA2)
end

function skew_mat_mul_test2(n, T=Float64)
    S = rand(SkewSymMatrix{T}, n)
    A = rand(n, n)
    AS1 = A*S 
    AS2 = A*Matrix{T}(S)
    @test isapprox(AS1, AS2)
end

# test Stiefel manifold projection test 
function stiefel_proj_test(N,n)
    In = I(n)
    E = StiefelProjection(N, n, Float64)
    @test all(abs.((E'*E) .- In) .< 1e-10)
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

function stiefel_lie_alg_vectorization_test(N, n; T=Float32)
    A = rand(StiefelLieAlgHorMatrix{T}, N, n)
    @test isapprox(StiefelLieAlgHorMatrix(vec(A), N, n), A)
end

# TODO: tests for ADAM functions

# test everything for different n & N values
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
    skew_mat_mul_test(N)
    skew_mat_mul_test2(N)
    stiefel_proj_test(N,n)
    stiefel_lie_alg_add_sub_test(N,n)
    stiefel_lie_alg_vectorization_test(N, n)
end
