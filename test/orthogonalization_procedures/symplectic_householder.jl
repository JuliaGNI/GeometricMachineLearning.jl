using GeometricMachineLearning
using Test

import LinearAlgebra

function rectangular_error(R::AbstractMatrix)
    N, M = size(R)
    #see how different R is from a "perfect rectangular matrix"
    diff = 0
    for j in 1:M
        for i in (j+1):N
            diff += abs(R[i,j])
        end
    end
    diff
end

function rectangular_error(R::GeometricMachineLearning.Rfac)
    N, M = size(R).÷2
    diff = 0 
    for i in 1:2
        for j in 1:2
            diff += rectangular_error(R[(1:N).+(i-1)*N, (1:M).+(j-1)*M])
        end
    end
    diff
end

function symplecticity_error(S::AbstractMatrix)
    N2, M2 = size(S)
    @assert iseven(N2)
    N, M = N2÷2, M2÷2
    @assert N == M 
    J = SymplecticPotential(N)
    LinearAlgebra.norm(S'*J*S - J)
end

function SR_test(N, M, tol)
    A = randn(2*N, 2*M)
    SR = sr(A)
    @test rectangular_error(SR.R) < tol
    @test symplecticity_error(SR.S)/(N*N) < tol
    @test typeof(inv(SR.S)) <: GeometricMachineLearning.Sfac 
    @test LinearAlgebra.norm(inv(SR.S)*A - SR.R)/(N*M) < tol
    @test LinearAlgebra.norm(A - SR.S*SR.R)/(N*M) < tol 
end

tol = 1e-10
N_max = 10

for N = 1:N_max
    for M = 1:N
        SR_test(N, M, tol)
    end
end