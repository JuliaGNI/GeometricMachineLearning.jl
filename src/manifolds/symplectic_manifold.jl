mutable struct SymplecticStiefelManifold{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
    function SymplecticStiefelManifold(A::AbstractMatrix)
        @assert iseven(size(A)[1])
        @assert iseven(size(A)[2])
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
    #this should draw a random element from U(SymplecticStiefelManifold) -> doesn't work rn; implement using retractions!
    function SymplecticStiefelManifold(N::Int,n::Int)
        @assert N ≥ n
        A = randn(2*N,2*n)
        JN = SymplecticMatrix(N)
        Jn = SymplecticMatrix(n)
        new{eltype(A), typeof(A)}(A*inv(sqrt(Jn*A'*JN'*A)))
    end
end


function check(A::SymplecticStiefelManifold)
    N = size(A)[1]÷2
    n = size(A)[2]÷2
    norm(A'*SymplecticMatrix(N)*A - SymplecticMatrix(n))
    #print("Test passed.\n") 
end

function rgrad(U::SymplecticStiefelManifold, e_grad::AbstractMatrix, J::AbstractMatrix)
    e_grad * (U' * U) + J * U * (e_grad' * J * U)
end