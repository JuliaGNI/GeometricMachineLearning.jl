"""
The `horizontal part` of `SymplecticLieAlgMatrix` is a matrix
| A  B  |
| C -A' |, where B and C are symmetric 

with the following additional requirements: 
| A₁ A₂ᵀ|	
| A₃ 0  | = A,

| B₁ B₂ᵀ|
| B₂ 0  | = B, 

| C₁ C₂ᵀ|
| C₂ 0  | = C.

The horizontal component is here taken with respect to the G-invariant metric (X₁,X₂) ↦ tr(X₁⁺X₂). (This metric is bi-invariant but degenerate.)

The first index is the row index, the second one the column index.

TODO: Make this a subtype of SymplecticLieAlgMatrix!!!!!
"""

#-> AbstractVecOrMat!!
mutable struct SymplecticLieAlgHorMatrix{T, AT <: AbstractMatrix{T}, ST <: SymmetricMatrix{T}} <: AbstractMatrix{T}
    A₁::AT
    A₂::AT
    A₃::AT
    B₁::ST
    B₂::AT
    C₁::ST
    C₂::AT
    N::Int
    n::Int

    function SymplecticLieAlgHorMatrix(A₁::AbstractMatrix, A₂::AbstractMatrix, A₃::AbstractMatrix, 
                                    B₁::SymmetricMatrix, B₂::AbstractMatrix, C₁::SymmetricMatrix, 
                                    C₂::AbstractMatrix, N::Int, n::Int)
        @assert eltype(A₁) == eltype(A₂) == eltype(A₃) == eltype(B₁) == eltype(B₂) == eltype(C₁) == eltype(C₂)
        @assert size(A₁)[1] == size(A₁)[2] == size(A₂)[2] == size(A₃)[2] == B₁.n == size(B₂)[2] == C₁.n == size(C₂)[2] == n
        @assert size(A₂)[1] == size(A₃)[1] == size(B₂)[1] == size(C₂)[1] == (N-n)
        new{eltype(A₁), typeof(A₁), typeof(B₁)}(A₁, A₂, A₃, B₁, B₂, C₁, C₂, N, n)
    end
    function SymplecticLieAlgHorMatrix(S::SymplecticLieAlgMatrix, n::Int)
        N = S.n
        new{eltype(S.A), typeof(S.A), typeof(S.B)}(
            S.A[1:n, 1:n],
            S.A[1:n, (n+1):N]',
            S.A[(n+1):N, 1:n],
            #this could be made more efficient by accessing the vector that parametrizes SymmetricMatrix!!
            SymmetricMatrix(S.B[1:n, 1:n]),
            S.B[(n+1):N, 1:n],
            SymmetricMatrix(S.C[1:n, 1:n]),
            S.C[(n+1):N, 1:n],
            N, 
            n
        )
    end

    function SymplecticLieAlgHorMatrix(S::πₑ)
        new{eltype(S.A), typeof(S.A), typeof(S.B)}(
            S.A,
            -S.H,
            S.C,
            S.B,
            S.D,
            S.E,
            S.G,
            S.N,
            S.n
        )
    end
end 


#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
function A_index(A₁::AbstractMatrix, A₂::AbstractMatrix, A₃::AbstractMatrix, n, i, j)
    if i ≤ n
        if j ≤ n 
            return A₁[i,j]
        end
        return A₂[j-n, i]
    end
    if j ≤ n
        return A₃[i - n, j]
    end
    return 0.
end

function BC_index(B₁::SymmetricMatrix, B₂::AbstractMatrix, n, i, j)
    if i ≤ n
        if j ≤ n 
            return B₁[i,j]
        end
        return B₂[j-n, i]
    end
    if j ≤ n
        return B₂[i - n, j]
    end
    return 0.
end

function Base.getindex(S::SymplecticLieAlgHorMatrix, i, j)
    N = S.N
    n = S.n

    if i ≤ N && j ≤ N
        return A_index(S.A₁, S.A₂, S.A₃, n, i, j)
    end
    if i ≤ N
        return BC_index(S.B₁, S.B₂, n, i, j-N)
    end
    if j ≤ N
        return BC_index(S.C₁, S.C₂, n, i-N, j)
    end
    return -A_index(S.A₁, S.A₂, S.A₃, n, j-N, i-N)
end


Base.parent(S::SymplecticLieAlgHorMatrix) = (A₁=S.A₁, A₂=S.A₂, A₃=S.A₃, B₁=S.B₁, B₂=S.B₂, C₁=S.C₁, C₂=S.C₂)
Base.size(S::SymplecticLieAlgHorMatrix) = (2*S.N,2*S.N)

function Base.:+(S₁::SymplecticLieAlgHorMatrix, S₂::SymplecticLieAlgHorMatrix) 
    @assert S₁.n == S₂.n  
    @assert S₁.N == S₂.N
    SymplecticLieAlgHorMatrix(
        S₁.A₁ + S₂.A₂,
        S₁.A₂ + S₂.A₂,
        S₁.A₃ + S₂.A₃,
        S₁.B₁ + S₂.B₁,
        S₁.B₂ + S₂.B₂,
        S₁.C₁ + S₂.C₁,
        S₁.C₂ + S₂.C₂,
        S₁.N, 
        S₁.n
        )
end

function Base.:-(S₁::SymplecticLieAlgHorMatrix, S₂::SymplecticLieAlgHorMatrix) 
    @assert S₁.n == S₂.n  
    @assert S₁.N == S₂.N
    SymplecticLieAlgHorMatrix(
        S₁.A₁ - S₂.A₂,
        S₁.A₂ - S₂.A₂,
        S₁.A₃ - S₂.A₃,
        S₁.B₁ - S₂.B₁,
        S₁.B₂ - S₂.B₂,
        S₁.C₁ - S₂.C₁,
        S₁.C₂ - S₂.C₂,
        S₁.N, 
        S₁.n
        )
end


#function Base.:./(A::SymplecticLieAlgMatrix,B::SymplecticLieAlgMatrix)
function Adam_div(S₁::SymplecticLieAlgHorMatrix,S₂::SymplecticLieAlgHorMatrix, δ=1e-8)
    @assert S₁.n == S₂.n
    @assert S₁.N == S₂.N
    SymplecticLieAlgMatrix(
        S₁.A₁/(S₂.A₁ .+ δ), 
        S₁.A₂/(S₂.A₂ .+ δ),
        S₁.A₃/(S₂.A₃ .+ δ),
        SymmetricMatrix(S₁.B₁.S/(S₂.B₁.S .+ δ),n), 
        S₁.B₂/(S₂.B₂ .+ δ),
        SymmetricMatrix(S₁.B₁.S/(S₂.B₁.S .+ δ),n), 
        S₁.C₂/(S₂.C₂ .+ δ),
        N,
        n
        )
end

function ⊙(S::SymplecticLieAlgHorMatrix)
    SymplecticLieAlgMatrix(
        S.A₁.^2, 
        S.A₂.^2,
        S.A₃.^2,
        SymmetricMatrix(S.B₁.S.^2,n), 
        S.B₂.^2,
        SymmetricMatrix(S.B₁.S.^2,n), 
        S.C₂.^2,
        N,
        n
        )
end

function Base.:√(S::SymplecticLieAlgHorMatrix) 
    SymplecticLieAlgMatrix(
        sqrt.(S.A₁), 
        sqrt.(S.A₂),
        sqrt.(S.A₃),
        SymmetricMatrix(sqrt.(S.B₁.S),n), 
        sqrt.(S.B₂),
        SymmetricMatrix(sqrt.(S.B₁.S),n), 
        sqrt.(S.C₂),
        N,
        n
        )
end