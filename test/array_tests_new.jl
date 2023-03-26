using LinearAlgebra
using Test

using BandedMatrices

"""
𝔤ʰ is the horizontal part of 𝔤, i.e. A ∈ 𝔤ʰ ⟺ AEE⁺ = A.

TODO: Add routine & test for symplectic conjugate
"""

include("../src/arrays/symmetric2.jl")
include("../src/arrays/symplectic_lie_alg2.jl")
include("../src/arrays/symplectic.jl")
include("../src/arrays/sympl_st_E_ts.jl")
include("../src/arrays/sympl_lie_alg_hor.jl")

#check if symmetric matrix works for 1×1 matrices 
W = rand(1,1)
S = SymmetricMatrix(W)
@test abs(W[1,1] - S[1,1]) < 1e-10

#check if matrix addition & subtraction works   
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
    for i in 1:n
        for j in 1:n
            @test abs(symmetrize(W₁ + W₂)[i,j] - S₃[i,j]) < 1e-10
            @test abs(symmetrize(W₁ - W₂)[i,j] - S₄[i,j]) < 1e-10
        end
    end 
end

#check if matrix is ∈ 𝔤
function sympl_lie_alg_test(N)
    W = rand(2*N, 2*N)
    JN = SymplecticMatrix(N)
    S = SymplecticLieAlgMatrix(W)
    for i in 1:(2*N)
        for j in 1:(2*N)
            @test abs(S[i,j] + (JN'*S'*JN)[i,j]) < 1e-10
        end 
    end 
end

#check if SymplecticLieAlgMatrix is closed under addition and subtraction
function sympl_lie_alg_add_sub_test(N)
    W₁ = rand(2*N,2*N)
    W₂ = rand(2*N,2*N)
    S₁ = SymplecticLieAlgMatrix(W₁)
    S₂ = SymplecticLieAlgMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SymplecticLieAlgMatrix
    @test typeof(S₄) <: SymplecticLieAlgMatrix
end

#test symplectic projection
function sympl_proj_test(N, n)
    JN = SymplecticMatrix(N)
    Jn = SymplecticMatrix(n)
    E = SymplecticProjection(N, n, Float64)
    for i in 1:(2*n)
        for j in 1:(2*n) 
            @test abs((E'*JN*E)[i,j] - Jn[i,j]) < 1e-10
        end
    end
end

#test horizontal lift of Lie Algebra
function hor_lift_test(N,n)
    E = SymplecticProjection(N, n, Float64)
    #element of 𝔤
    S = SymplecticLieAlgMatrix(rand(2*N, 2*N))
    #compute projection onto 𝔤ʰ
    Sʰ = SymplecticLieAlgHorMatrix(S,n)
    #test projection
    for i in 1:(2*N)
        for j in 1:(2*N)
            #compute projection
            πS = πₑ(S*E)
            #compute lift
            πS_lifted = SymplecticLieAlgHorMatrix(πS)
            @test abs(πS_lifted[i,j] - Sʰ[i,j]) < 1e-10
        end
    end
end

#TODO: tests for ADAM functions


#test everything for different n & N values
N_max = 20
n_max = 10
num = 100
N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)

for (N, n) ∈ zip(N_vec, n_vec)
    sym_mat_add_sub_test(N)
    sympl_lie_alg_test(N)
    sympl_lie_alg_add_sub_test(N)
    sympl_proj_test(N,n)
    hor_lift_test(N,n)
end