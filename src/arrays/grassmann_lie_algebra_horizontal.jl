@doc raw"""
    GrassmannLieAlgHorMatrix(B::AbstractMatrix, N::Integer, n::Integer)

Build an instance of `GrassmannLieAlgHorMatrix` based on an arbitrary matrix `B` of size ``(N-n)\times{}n``.

`GrassmannLieAlgHorMatrix` is the *horizontal component of the Lie algebra of skew-symmetric matrices* (with respect to the canonical metric).

# Extended help

The projection here is: ``\pi:S \to SE/\sim`` where 
```math
E = \begin{bmatrix} \mathbb{I}_{n} \\ \mathbb{O}_{(N-n)\times{}n}  \end{bmatrix},
```

and the equivalence relation is 

```math
V_1 \sim V_2 \iff \exists A\in\mathcal{S}_\mathrm{skew}(n) \text{ such that } V_2 = V_1 + \begin{bmatrix} A \\ \mathbb{O} \end{bmatrix}
```

An element of GrassmannLieAlgMatrix takes the form: 
```math
\begin{pmatrix}
\bar{\mathbb{O}} & B^T \\ B & \mathbb{O}
\end{pmatrix},
```
where ``\bar{\mathbb{O}}\in\mathbb{R}^{n\times{}n}`` and ``\mathbb{O}\in\mathbb{R}^{(N - n)\times(N-n)}.``
"""
mutable struct GrassmannLieAlgHorMatrix{T, ST <: AbstractMatrix{T}} <: AbstractLieAlgHorMatrix{T}
    B::ST
    N::Int
    n::Int 

    #maybe modify this - you don't need N & n as inputs!
    function GrassmannLieAlgHorMatrix(B::AbstractMatrix{T}, N::Int, n::Int) where {T}
        @assert n == size(B,2) 
        @assert N == size(B,1) + n

        new{T, typeof(B)}(B, N, n)
    end 
end 

@doc raw"""
    GrassmannLieAlgHorMatrix(D::AbstractMatrix, n::Integer)

Take a big matrix as input and build an instance of `GrassmannLieAlgHorMatrix`.

The integer ``N`` in ``Gr(n, N)`` here is the number of rows of `D`.

# Extended help

If the constructor is called with a big ``N\times{}N`` matrix, then the projection is performed the following way: 

```math
\begin{pmatrix}
A & B_1  \\
B_2 & D
\end{pmatrix} \mapsto 
\begin{pmatrix}
\bar{\mathbb{O}} & -B_2^T \\ 
B_2 & \mathbb{O}
\end{pmatrix}.
```

This can also be seen as the operation:
```math
D \mapsto \Omega(E, DE - EE^TDE),
```

where ``\Omega`` is the horizontal lift [`GeometricMachineLearning.Ω`](@ref).
"""
function GrassmannLieAlgHorMatrix(D::AbstractMatrix, n::Int)
    N = size(D, 1)
    @assert N ≥ n 

    @views B = D[(n + 1):N,1:n]
    GrassmannLieAlgHorMatrix(B, N, n)
end

Base.parent(A::GrassmannLieAlgHorMatrix) = (A.B, )
Base.size(A::GrassmannLieAlgHorMatrix) = (A.N, A.N)

KernelAbstractions.get_backend(B::GrassmannLieAlgHorMatrix) = KernelAbstractions.get_backend(B.B)

function Base.getindex(A::GrassmannLieAlgHorMatrix{T}, i::Integer, j::Integer) where {T}
    if i ≤ A.n
        if j ≤ A.n 
            return T(0.)
        end
        return -A.B[j - A.n, i]
    end
    if j ≤ A.n 
        return A.B[i - A.n, j]
    end
    return T(0.)
end

function Base.:+(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    GrassmannLieAlgHorMatrix(A.B + B.B, 
                            A.N,
                            A.n)
end

function Base.:-(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    GrassmannLieAlgHorMatrix(A.B - B.B, 
                            A.N,
                            A.n)
end

function add!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N == C.N
    @assert A.n == B.n == C.n 
    add!(C.B, A.B, B.B)  
end

function Base.:-(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix( -A.B, A.N, A.n)
end

function Base.:*(A::GrassmannLieAlgHorMatrix, α::Real)
    GrassmannLieAlgHorMatrix( α*A.B, A.N, A.n)
end

Base.:*(α::Real, A::GrassmannLieAlgHorMatrix) = A*α

function Base.zeros(::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    GrassmannLieAlgHorMatrix(
        zeros(T, N-n, n),
        N, 
        n
    )
end
    
function Base.zeros(::Type{GrassmannLieAlgHorMatrix}, N::Integer, n::Integer)
    GrassmannLieAlgHorMatrix(
        zeros(N-n, n),
        N, 
        n
    )
end

function Base.zeros(backend::KernelAbstractions.Backend, ::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T 
    GrassmannLieAlgHorMatrix(
        KernelAbstractions.zeros(backend, T, N-n, n),
        N, 
        n
    )
end

Base.similar(A::GrassmannLieAlgHorMatrix, dims::Union{Integer, AbstractUnitRange}...) = zeros(typeof(A), dims...)
Base.similar(A::GrassmannLieAlgHorMatrix) = zeros(typeof(A), A.N, A.n)

function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    GrassmannLieAlgHorMatrix(rand(rng, T, N-n, n), N, n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmannLieAlgHorMatrix}, N::Integer, n::Integer)
    GrassmannLieAlgHorMatrix(rand(rng, N-n, n), N, n)
end

function Base.rand(::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    rand(Random.default_rng(), GrassmannLieAlgHorMatrix{T}, N, n)
end

function Base.rand(::Type{GrassmannLieAlgHorMatrix}, N::Integer, n::Integer)
    rand(Random.default_rng(), GrassmannLieAlgHorMatrix, N, n)
end

function scalar_add(A::GrassmannLieAlgHorMatrix, δ::Real)
    GrassmannLieAlgHorMatrix(A.B .+ δ, A.N, A.n)
end

#define these functions more generally! (maybe make a fallback script!!)
function ⊙²(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(A.B.^2, A.N, A.n)
end
function racᵉˡᵉ(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(sqrt.(A.B), A.N, A.n)
end
function /ᵉˡᵉ(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(A.B./B.B, A.N, A.n)
end 

function LinearAlgebra.mul!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix, α::Real)
    mul!(C.B, A.B, α)
end
LinearAlgebra.mul!(C::GrassmannLieAlgHorMatrix, α::Real, A::GrassmannLieAlgHorMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::GrassmannLieAlgHorMatrix, α::Real) = mul!(C, C, α)

function _round(B::GrassmannLieAlgHorMatrix; kwargs...)
    GrassmannLieAlgHorMatrix(
        _round(B.B; kwargs...),
        B.N,
        B.n
    )
end