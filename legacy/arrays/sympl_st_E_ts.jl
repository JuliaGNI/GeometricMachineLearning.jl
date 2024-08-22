"""
Implements the tangent space of Sp(2n,2N) =: 𝑍 at the point E (defined in "poisson_tensor.jl").
This tangent space is defined through Tₑ𝑍 = {Δ:Δ⁺E + E⁺Δ = 0}, where ⁺:Δ ↦ JΔᵀJᵀ is the symplectic conjugation.
It is also isomorphic to the space 𝔤ʰ (see "symplectic_lie_alg_hor.jl").

Specifically, it takes the following form:
|A  B |
|C  D |
|E -Aᵀ|
|G  H | = Δ, where B and E are required to be symmetric.

The isomorphism with 𝔤ʰ works through:
A₁ →  A,
A₂ → -H,
A₃ →  C,
B₁ →  B,
B₂ →  D,
C₁ →  E,
C₂ →  G.

The first index is the row index, the second one the column index.
"""

#-> AbstractVecOrMat!!
mutable struct πₑ{T, AT <: AbstractMatrix{T}, ST <: SymmetricMatrix{T}} <: AbstractMatrix{T}
    A::AT
    H::AT
    C::AT
    B::ST
    D::AT
    E::ST
    G::AT
    N::Int
    n::Int

    function πₑ( A::AbstractMatrix, H::AbstractMatrix, C::AbstractMatrix, 
                                        B::SymmetricMatrix, D::AbstractMatrix, E::SymmetricMatrix, 
                                        G::AbstractMatrix, N::Int, n::Int)
        @assert eltype(A) == eltype(B) == eltype(C) == eltype(D) == eltype(E) == eltype(G) == eltype(H)
        @assert size(A)[1] == size(A)[2] == B.n == size(C)[2] == size(D)[2] == E.n == size(G)[2] == size(H)[2] == n
        @assert size(C)[1] == size(D)[1] == size(G)[1] == size(H)[1] == (N-n)
        new{eltype(A), typeof(A), typeof(B)}(A, H, C, B, D, E, G, N, n)
    end
    function πₑ(S::SymplecticLieAlgMatrix, n::Int)
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
    #fix this chicken-or-egg problem!!!!!
    #=
    function πₑ(S::SymplecticLieAlgHorMatrix)
        new{eltype(S.A₂), typeof(S.A₁), typeof(S.B₁)}(
            S.A₁,
            -S.A₂, 
            S.A₃, 
            S.B₁,
            S.B₂,
            S.C₁,
            S.C₂,
            N,
            n
        )
    end
    =#
    function πₑ(S::AbstractMatrix)
        @assert iseven(size(S)[1])
        @assert iseven(size(S)[2])
        N = size(S)[1]÷2
        n = size(S)[2]÷2
        new{eltype(S), AbstractMatrix{eltype(S)}, SymmetricMatrix{eltype(S)}}(
            S[1:n, 1:n],
            S[(N+n+1):(2*N), (n+1):(2*n)],
            S[(n+1):N, 1:n],
            SymmetricMatrix(S[1:n, (n+1):(2*n)]),
            S[(n+1):N, (n+1):(2*n)],
            SymmetricMatrix(S[(N+1):(N+n), 1:n]),
            S[(N+n+1):(2*N), 1:n],
            N,
            n
        )
    end
end 


#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
function Base.getindex(S::πₑ, i, j)
    N = S.N
    n = S.n
    if i ≤ n 
        if j ≤ n
            return S.A[i,j]
        end
        return S.B[i,j-n]
    end
    if i ≤ N
        if j ≤ n
            return S.C[i-n, j]
        end
        return S.D[i-n, j-n]
    end
    if i ≤ (N+n)
        if j ≤ n
            return S.E[i-N, j]
        end
        return -S.A[j-n, i-N]
    end
    if j ≤ n 
        return S.G[i-(N+n), j]
    end
    return S.H[i-(N+n), j-n]
end


Base.parent(S::πₑ) = (A=S.A, H=S.H, C=S.C, B=S.B, D=S.D, E=S.E, G=S.G)
Base.size(S::πₑ) = (2*S.N,2*S.n)

function Base.:+(S₁::πₑ, S₂::πₑ) 
    @assert S₁.n == S₂.n  
    @assert S₁.N == S₂.N
    πₑ(
        S₁.A + S₂.A,
        S₁.H + S₂.H,
        S₁.C + S₂.C,
        S₁.B + S₂.B,
        S₁.D + S₂.D,
        S₁.E + S₂.E,
        S₁.G + S₂.G,
        S₁.N, 
        S₁.n
        )
end

function Base.:-(S₁::πₑ, S₂::πₑ) 
    @assert S₁.n == S₂.n  
    @assert S₁.N == S₂.N
    πₑ(
        S₁.A - S₂.A,
        S₁.H - S₂.H,
        S₁.C - S₂.C,
        S₁.B - S₂.B,
        S₁.D - S₂.D,
        S₁.E - S₂.E,
        S₁.G - S₂.G,
        S₁.N, 
        S₁.n
        )
end

#implement these functions!
#=
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

function Base.:RAC(S::SymplecticLieAlgHorMatrix) 
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
=#