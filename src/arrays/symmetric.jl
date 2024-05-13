@doc raw"""
A `SymmetricMatrix` ``A`` is a matrix ``A^T = A``.

This is a projection defined via the canonical metric $(A,B) \mapsto \mathrm{tr}(A^TB)$.

Internally the `struct` saves a vector $S$ of size $n(n+1)\div2$. The conversion is done the following way: 
```math
[A]_{ij} = \begin{cases} S[( (i-1) i ) \div 2 + j] & \text{if $i\geq{}j$}\\ 
                         S[( (j-1) j ) \div 2 + i] & \text{else}. \end{cases}
```

So ``S`` stores a string of vectors taken from $A$: $S = [\tilde{a}_1, \tilde{a}_2, \ldots, \tilde{a}_n]$ with $\tilde{a}_i = [[A]_{i1},[A]_{i2},\ldots,[A]_{ii}]$.

### Constructor 
If the constructor is called with a matrix as input it returns a symmetric matrix via the projection:
```math
A \mapsto \frac{1}{2}(A + A^T).
```

It can also be called with two arguments `S::AbstractVector` and `n::Integer` where `length(S) == n * (n + 1) ÷ 2` has to be true.
"""
mutable struct SymmetricMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int

    function SymmetricMatrix(S::AbstractVector, n::Integer)
        @assert length(S) == n*(n+1)÷2
        new{eltype(S),typeof(S)}(S, n)
    end
    function SymmetricMatrix(A::AbstractMatrix{T}) where {T}
        S = map_to_S(A)
        new{T, typeof(S)}(S, size(A, 1))
    end
end 

# I'm not 100% sure this is the best solution (needed for broadcasting operations ...)
function Base.setindex!(A::SymmetricMatrix{T}, val::T, i::Int, j::Int) where T
    if i ≥ j 
        A.S[i * (i-1)÷2 + j] = val 
    else
        A.S[j * (j-1)÷2 + i] = val 
    end
end

@kernel function assign_S_val_kernel!(S, A_sym, i)
    j = @index(Global)
    S[i * (i-1)÷2 + j] = A_sym[i, j]
end

function map_to_S(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    @assert size(A, 2) == n 
    A_sym = T(.5)*(A + A')
    backend = KernelAbstractions.get_backend(A)
    S = KernelAbstractions.zeros(backend, T, n*(n+1)÷2)
    assign_S_val! = assign_S_val_kernel!(backend)
    for i in 1:n
        assign_S_val!(S, A_sym, i, ndrange=i)
    end
    S
end

function LinearAlgebra.Adjoint(A::SymmetricMatrix)
    A 
end

function Base.zero(A::SymmetricMatrix)
    SymmetricMatrix(zero(A.S), A.n)
end

function Base.getindex(A::SymmetricMatrix,i::Int,j::Int)
    if i ≥ j
        A.S[((i-1)*i)÷2+j]
    else
        A.S[(j-1)*j÷2+i]
    end
end

Base.parent(A::SymmetricMatrix) = A.S
Base.size(A::SymmetricMatrix) = (A.n,A.n)

function Base.:+(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n 
    SymmetricMatrix(A.S + B.S, A.n) 
end 

function add!(C::SymmetricMatrix, A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function Base.:-(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n 
    SymmetricMatrix(A.S - B.S, A.n) 
end 


function Base.:-(A::SymmetricMatrix)
    SymmetricMatrix(-A.S, A.n)
end

function Base.:*(A::SymmetricMatrix, α::Real)
    SymmetricMatrix(α*A.S, A.n)
end

Base.:*(α::Real, A::SymmetricMatrix) = A*α

function Base.zeros(::Type{SymmetricMatrix{T}}, n::Int) where T
    SymmetricMatrix(zeros(T, n*(n+1)÷2), n)
end
    
function Base.zeros(::Type{SymmetricMatrix}, n::Int)
    SymmetricMatrix(zeros(n*(n+1)÷2), n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymmetricMatrix{T}}, n::Int) where T
    SymmetricMatrix(rand(rng, T, n*(n+1)÷2),n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymmetricMatrix}, n::Int)
    SymmetricMatrix(rand(rng, n*(n+1)÷2), n)
end

#TODO: make defaults when no rng is specified!!! (prbabaly rng ← Random.default_rng())
function Base.rand(type::Type{SymmetricMatrix{T}}, n::Integer) where T
    rand(Random.default_rng(), type, n)
end

function Base.rand(type::Type{SymmetricMatrix}, n::Integer)
    rand(Random.default_rng(), type, n)
end

#these are Adam operations:
function scalar_add(A::SymmetricMatrix, δ::Real)
    SymmetricMatrix(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::SymmetricMatrix)
    SymmetricMatrix(A.S.^2, A.n)
end
function racᵉˡᵉ(A::SymmetricMatrix)
    SymmetricMatrix(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n 
    SymmetricMatrix(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::SymmetricMatrix, A::SymmetricMatrix, α::Real)
    mul!(C.S, A.S, α)
end
LinearAlgebra.mul!(C::SymmetricMatrix, α::Real, A::SymmetricMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::SymmetricMatrix, α::Real) = mul!(C, C, α)

@kernel function symmetric_mat_mul_kernel!(C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where T 
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = 1:i 
        tmp_sum += S[((i-1)*i)÷2+k] * B[k, j]
    end
    for k = (i+1):n 
        tmp_sum += S[((k-1)*k)÷2+i] * B[k, j]
    end
    C[i, j] = tmp_sum
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricMatrix, B::AbstractMatrix)
    @assert A.n == size(B, 1)
    @assert size(B, 2) == size(C, 2)
    @assert A.n == size(C, 1)
    backend = KernelAbstractions.get_backend(A.S)
    symmetric_mat_mul! = symmetric_mat_mul_kernel!(backend)
    symmetric_mat_mul!(C, A.S, B, A.n, ndrange=size(C))
end

@kernel function symmetric_vector_mul_kernel!(c::AbstractVector{T}, S::AbstractVector{T}, b::AbstractVector{T}, n) where T 
    i = @index(Global)

    tmp_sum = zero(T)
    for k = 1:i
        tmp_sum += S[((i-1)*i)÷2+k] * b[k]
    end
    for k = (i+1):n 
        tmp_sum += S[((k-1)*k)÷2+i] * b[k]
    end
    c[i] = tmp_sum
end

function LinearAlgebra.mul!(c::AbstractVector, A::SymmetricMatrix, b::AbstractVector)
    @assert A.n == length(c) == length(b)
    backend = KernelAbstractions.get_backend(A.S)
    symmetric_vector_mul! = symmetric_vector_mul_kernel!(backend)
    symmetric_vector_mul!(c, A.S, b, A.n, ndrange=size(c))
end

function Base.:*(A::SymmetricMatrix{T}, B::AbstractMatrix{T}) where T
    backend = KernelAbstractions.get_backend(A.S)
    C = KernelAbstractions.allocate(backend, T, A.n, size(B, 2))
    LinearAlgebra.mul!(C, A, B)
    C
end

Base.:*(B::AbstractMatrix{T}, A::SymmetricMatrix{T}) where T = (A * B')'

function Base.:*(A::SymmetricMatrix{T}, B::SymmetricMatrix{T}) where T 
    A * (B * one(B))
end

function Base.:*(A::SymmetricMatrix{T}, b::AbstractVector{T}) where T 
    backend = KernelAbstractions.get_backend(A.S)
    c = KernelAbstractions.allocate(backend, T, A.n)
    LinearAlgebra.mul!(c, A, b)
    c
end

function Base.one(A::SymmetricMatrix{T}) where T
    backend = KernelAbstractions.get_backend(A.S)
    unit_matrix = KernelAbstractions.zeros(backend, T, A.n, A.n)
    write_ones! = write_ones_kernel!(backend)
    write_ones!(unit_matrix, ndrange=A.n)
    unit_matrix
end

# define routines for generalizing ChainRulesCore to SymmetricMatrix 
ChainRulesCore.ProjectTo(A::SymmetricMatrix) = ProjectTo{SymmetricMatrix}(; symmetric=ProjectTo(A.S))
(project::ProjectTo{SymmetricMatrix})(dA::AbstractMatrix) = SymmetricMatrix(project.symmetric(map_to_S(dA)), size(dA, 2))
(project::ProjectTo{SymmetricMatrix})(dA::SymmetricMatrix) = SymmetricMatrix(project.symmetric(dA.S), dA.n)