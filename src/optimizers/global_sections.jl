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

#start index indicates if the orthonormalization is started at positon 0
function gram_schmidt!(A::AbstractMatrix, start=1)
    n = size(A)[1]
    @assert size(A)[2] == n 
    
    for i in start:n 
        vec = A[1:n,i]
        for j in 1:(i-1)
            vec = vec - vec'*A[1:N,j]*A[1:N,j]
        end
        A[1:N, i] = norm(vec)^-1*vec 
    end
end 

function sympl_gram_schmidt!(A::AbstractMatrix, J::AbstractMatrix, start=1)
    n = size(A)[1]
    @assert size(A)[2] == n 
    @assert iseven(n)
    n ÷= 2

    for i in start:n 
        vec₁ = A[1:(2*n),i]
        vec₂ = A[1:(2*n),n+i]
        for j in 1:(i-1)
            vec₁ = vec₁ - (A[1:(2*n),j]'*J*vec₁)*A[1:(2*n),j] - (vec₁'*J*A[1:(2*n),n+j])*A[1:(2*n),n+j]
            vec₂ = vec₂ - (A[1:(2*n),j]'*J*vec₂)*A[1:(2*n),j] - (vec₂'*J*A[1:(2*n),n+j])*A[1:(2*n),n+j]
        end
        A[1:(2*n),i] =      norm(vec₁)^-1*vec₁
        A[1:(2*n),n+i] =    norm(vec₂)^-1*vec₂ 
    end
end 

function gram_schmidt(A::AbstractMatrix, start=1)
    B = deepcopy(A)
    gram_schmidt!(B, start)
    B
end

function sympl_gram_schmidt(A::AbstractMatrix, J::AbstractMatrix, start=1)
    B = deepcopy(A)
    sympl_gram_schmidt!(B, J, start)
    B
end

#orthonormal complection -> the complementing vectors could also be sampled!!
function global_section(A::StiefelManifold)
    N = size(A)[1]
    n = size(A)[2]

    completed_A = zeros(N,N)
    
    for i in 1:n 
        completed_A[1:N, i] = A[1:N, i]
    end

    for i in (n+1):N 
        completed_A[i,i] = 1.
    end
    
    gram_schmidt!(completed_A, n+1)
    completed_A
end

function global_section(A::SymplecticStiefelManifold, J::AbstractMatrix)
    N = size(A)[1]÷2
    n = size(A)[2]÷2

    completed_A = zeros(N,N)

    for i in 1:n
        completed_A[1:(2*N), i] = A[1:(2*N), i]
        completed_A[1:(2*N), N+i] = A[1:(2*N), n+i]
    end

    for i in (n+1):N 
        completed_A[i, i] = 1.
        completed_A[N+i,N+i] = 1.
    end

    sympl_gram_schmidt!(completed_A, J, n+1)
    completed_A
end

####Maybe put this in a separate file! (some of the functionality between the group and the StMan is the same! -> combine!!!)
mutable struct OrthonormalMatrix{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
    function OrthonormalMatrix(A::AbstractMatrix, orth=false)
        if orth ==true 
            new{eltype(A), typeof(A)}(A)
        end
        new{eltype(A), typeof(A)}(gram_schmidt(A))    
    end
end

Base.size(A::OrthonormalMatrix) = size(A.A)
Base.parent(A::OrthonormalMatrix) = A.A 
Base.getindex(A::OrthonormalMatrix, i::Int, j::Int) = A.A[i,j]

function check(A::OrthonormalMatrix, tol=1e-12)
    @assert norm(A'*A - I) < tol
    print("Test passed.\n") 
end