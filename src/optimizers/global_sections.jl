"""
This implements global sections for the Stiefel manifold and the Symplectic Stiefel manifold. 

In practice this is implemented through the Gram Schmidt process, with the auxiliary column vectors given by: 
|0|
|0|
|.|
|1| ith spot for i in (n+1) to N 
|0|
|.|
|0|
"""

mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
    function StiefelManifold(A::AbstractMatrix)
        @assert size(A)[1] > size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
end

Base.size(A::StiefelManifold) = size(A.A)
Base.parent(A::StiefelManifold) = A.A 
Base.getindex(A::StiefelManifold, i::Int, j::Int) = A.A[i,j]


function check(A::StiefelManifold, tol=1e-12)
    @assert norm(A'*A - I) < tol
    print("Test passed.\n") 
end

#orthonormal complection
function global_section(A::StiefelManifold)
    N = size(A)[1]
    n = size(A)[2]

    completed_A = zeros(N,N)
    
    for i in 1:n 
        completed_A[1:N, i] = A[1:N, i]
    end
    
    for i in (n+1):N
        vec = zeros(N); vec[i] = 1.
        for j in 1:(i-1)
            vec = vec - vec'*completed_A[1:N,j]*completed_A[1:N,j]
        end
        completed_A[1:N, i] = norm(vec)^-1*vec 
    end
    completed_A
end


####Maybe put this in a separate file! (some of the functionality between the group and the StMan is the same! -> combine!!!)
mutable struct OrthonormalMatrix{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
    function OrthonormalMatrix(A::AbstractMatrix, orth=false)
        if orth ==true 
            new{eltype(A), typeof(A)}(A)
        end
        N  = size(A)[1]
        @assert size(A)[2] == N

        A_orthon = zeros(N, N)

        for i in 1:N
            vec = A[1:N, i]
            for j in 1:(i-1)
                vec = vec - vec'*A_orthon[1:N,j]*A_orthon[1:N,j]
            end
            A_orthon[1:N, i] = norm(vec)^-1*vec 
        end
        new{eltype(A), typeof(A)}(A_orthon)    
    end
end

Base.size(A::OrthonormalMatrix) = size(A.A)
Base.parent(A::OrthonormalMatrix) = A.A 
Base.getindex(A::OrthonormalMatrix, i::Int, j::Int) = A.A[i,j]

function check(A::OrthonormalMatrix, tol=1e-12)
    @assert norm(A'*A - I) < tol
    print("Test passed.\n") 
end