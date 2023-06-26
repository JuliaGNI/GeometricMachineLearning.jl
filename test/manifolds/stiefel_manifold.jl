using Test 
using LinearAlgebra
using GeometricMachineLearning

N = 5
A = rand(N,N)
A_skew = SkewSymMatrix(A)

for i in 1:N 
    for j in 1:N 
        @test abs(.5*(A - A')[i,j] - A_skew[i,j]) < 1e-10
    end
end

n = 1
A_hor = StiefelLieAlgHorMatrix(A_skew, n)

for i in 1:n
    for j in 1:N 
        @test abs(A_hor[i,j] - 2*A_skew[i,j]) < 1e-10
    end 
end

for i in (n+1):N 
    for j in 1:n 
        @test abs(A_hor[i,j] - A_skew[i,j]) < 1e-10
    end
    for j in (n+1):N 
        @test abs(A_hor[i,j]) < 1e-10
    end
end

#=
#Stiefel manifold test
A_ortho = OrthonormalMatrix(A)
check(A_ortho)
A_stiefel = A_ortho[1:N,1:n]
A_stiefel = StiefelManifold(A_stiefel)
check(A_stiefel)

A_ortho2 = global_section(A_stiefel)
A_ortho2 = OrthonormalMatrix(A_ortho2, true)
check(A_ortho2)

display(A_ortho)
display(A_ortho2)
=#