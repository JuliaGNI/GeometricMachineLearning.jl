"""
This tests the global sections that map into the Lie groups of orthonormal matrices and symplectic matrices as well as the GS procedures.
"""

using LinearAlgebra
using BandedMatrices
using Test

include("../src/optimizers/global_sections.jl")
include("../src/arrays/symplectic.jl")

N = 10
n = 5

function gram_schmidt_test(N)
    A = randn(N, N)
    A_orth = gram_schmidt(A)
    @test norm(A_orth'*A_orth - I) < 1e-10
end

function sympl_gram_schmidt_test(N)
    J = SymplecticMatrix(N)
    A = randn(2*N, 2*N)
    A_sympl = sympl_gram_schmidt(A, J)
    @test norm(A_sympl'*J*A_sympl - J) < 1e-10
end

function stiefel_completion_test(N,n)
    A = StiefelManifold(N,n)
    @test check(A)
    A_compl = global_section(A)
    @test check(StiefelManifold(A_compl))
end



#test everything for different n & N values
N_max = 20
n_max = 10
num = 100
N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)

for (N, n) ∈ zip(N_vec, n_vec)
    gram_schmidt_test(N)
    sympl_gram_schmidt_test(N)
end