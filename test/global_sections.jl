"""
This tests the global sections that map into the Lie groups of orthonormal matrices and symplectic matrices as well as the GS procedures.

TODO: for now the symplectic matrices are generated in a hacky way!!! Fix this in struct SymplecticStiefelManifold!!!!
"""

using LinearAlgebra
using BandedMatrices
using Test

include("../src/optimizers/orthogonalization_procedures.jl")
include("../src/optimizers/global_sections.jl")
include("../src/arrays/symplectic.jl")
include("../src/arrays/skew_sym.jl")
include("../src/arrays/symmetric2.jl")
include("../src/arrays/symplectic_lie_alg2.jl")
include("../src/optimizers/retractions.jl")

function stiefel_completion_test(N,n)
    A = StiefelManifold(N,n)
    @test check(A)
    A_compl = global_section(A)
    check(StiefelManifold(A_compl))
end

function symplectic_stiefel_completion_test(N,n)
    J = SymplecticMatrix(N)
    E = SymplecticProjection(N,n)
    A = randn(2*N,2*N)
    U = Cayley(SymplecticLieAlgMatrix(A))*E
    U = SymplecticStiefelManifold(U)
    U_compl = global_section(A, J)
    check(U_compl)
end


#test everything for different n & N values
N_max = 100
n_max = 10
num = 100
N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)

for (N, n) ∈ zip(N_vec, n_vec)
    #print(N,"\n")
    gram_schmidt_test(N)
    sympl_gram_schmidt_test(N)
end