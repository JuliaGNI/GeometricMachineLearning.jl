"""
A `BlockIdentityUpperMatrix` is a matrix with blocks
| 1  S |
| 0  1 |
Currently, it only implements a custom `mul!` method, exploiting this structure.
"""
struct BlockIdentityUpperMatrix{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    S::AT

    function BlockIdentityUpperMatrix(S::AbstractMatrix)
        @assert length(axes(S,1)) == length(axes(S,2))
        new{eltype(S), typeof(S)}(S)
    end
end

SymmetricBlockIdentityUpperMatrix(W::AbstractMatrix) = BlockIdentityUpperMatrix(W .+ W')

Base.parent(A::BlockIdentityUpperMatrix) = A.S

function LinearAlgebra.mul!(out::AbstractVector, A::BlockIdentityUpperMatrix, z::AbstractVector)
    @assert length(out) == length(z) == 2*length(axes(A.S,2))

    N = length(axes(A.S,1))
    
    q_in = @view z[1:N]
    p_in = @view z[N+1:2N]
    q_out = @view out[1:N]
    p_out = @view out[N+1:2N]

    mul!(q_out, A.S, p_in)

    q_out .+= q_in
    p_out  .= p_in

    return out
end
