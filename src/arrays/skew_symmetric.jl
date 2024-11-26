@doc raw"""
    SkewSymMatrix(S::AbstractVector, n::Integer)

Instantiate a skew-symmetric matrix with information stored in vector `S`.

A skew-symmetric matrix ``A`` is a matrix ``A^T = -A``.

Internally the `struct` saves a vector ``S`` of size ``n(n-1)\div2``. The conversion is done the following way: 
```math
[A]_{ij} = \begin{cases} 0                             & \text{if $i=j$} \\
                         S[( (i-2) (i-1) ) \div 2 + j] & \text{if $i>j$}\\ 
                         S[( (j-2) (j-1) ) \div 2 + i] & \text{else}. \end{cases}
```

So ``S`` stores a string of vectors taken from ``A``: ``S = [\tilde{a}_1, \tilde{a}_2, \ldots, \tilde{a}_n]`` with ``\tilde{a}_i = [[A]_{i1},[A]_{i2},\ldots,[A]_{i(i-1)}]``.

Also see [`SymmetricMatrix`](@ref), [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref).

# Examples 
```jldoctest
using GeometricMachineLearning
S = [1, 2, 3, 4, 5, 6]
SkewSymMatrix(S, 4)

# output

4×4 SkewSymMatrix{Int64, Vector{Int64}}:
 0  -1  -2  -4
 1   0  -3  -5
 2   3   0  -6
 4   5   6   0
```
"""
mutable struct SkewSymMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int

    function SkewSymMatrix(S::AbstractVector{T},n::Int) where {T}
        @assert length(S) == n*(n-1)÷2
        new{T,typeof(S)}(S,n)
    end
end 

@doc raw"""
    SkewSymMatrix(A::AbstractMatrix)

Perform `0.5 * (A - A')` and store the matrix in an efficient way (as a vector with ``n(n-1)/2`` entries).

If the constructor is called with a matrix as input it returns a skew-symmetric matrix via the projection:
```math
A \mapsto \frac{1}{2}(A - A^T).
```

# Examples
```jldoctest
using GeometricMachineLearning
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
SkewSymMatrix(M)

# output

4×4 SkewSymMatrix{Float64, Vector{Float64}}:
 0.0  -1.5  -3.0  -4.5
 1.5   0.0  -1.5  -3.0
 3.0   1.5   0.0  -1.5
 4.5   3.0   1.5   0.0
```

# Extended help

Note that the constructor is designed in such a way that it always returns matrices of type `SkewSymMatrix{<:AbstractFloat}` when called with a matrix, even if this matrix is of type `AbstractMatrix{<:Integer}`.

If the user wishes to allocate a matrix `SkewSymMatrix{<:Integer}` then call:

```julia
SkewSymMatrix(::AbstractVector, n::Integer)
```

Note that this is different from [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref) as no porjection takes place there.
"""
function SkewSymMatrix(S::AbstractMatrix{T}) where {T}
    n = size(S, 1)
    @assert size(S, 2) == n
    S_vec = map_to_Skew(S)
    SkewSymMatrix(S_vec, n)
end

function return_element(S::AbstractVector{T}, i::Int, j::Int) where T
    if j == i
        return zero(T)
    end
    if i > j
        return S[(i-2) * (i-1) ÷ 2 + j]
    end
    return - S[ (j-2) * (j-1) ÷ 2 + i]
end

function Base.getindex(A::SkewSymMatrix, i::Int, j::Int)
    return_element(A.S, i, j)
end


Base.parent(A::SkewSymMatrix) = A.S
Base.size(A::SkewSymMatrix) = (A.n,A.n)

@kernel function addition_kernel!(C::AbstractMatrix, S::AbstractVector, B::AbstractMatrix)
    i, j = @index(Global, NTuple)
    C[i, j] = return_element(S, i, j) + B[i, j]
end

function Base.:+(A::SkewSymMatrix{T}, B::AbstractMatrix{T}) where T
    @assert size(A) == size(B)
    backend = KernelAbstractions.get_backend(B)
    addition! = addition_kernel!(backend)
    C = KernelAbstractions.allocate(backend, T, size(A)...)
    addition!(C, A.S, B; ndrange = size(A))

    C
end

Base.:+(B::AbstractMatrix, A::SkewSymMatrix) = B + A

function Base.:+(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n 
    SkewSymMatrix(A.S + B.S, A.n) 
end 

function add!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function Base.:-(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n 
    SkewSymMatrix(A.S - B.S, A.n) 
end 

function Base.:-(A::SkewSymMatrix)
    SkewSymMatrix(-A.S, A.n)
end

function Base.:*(A::SkewSymMatrix, α::Real)
    SkewSymMatrix(α*A.S, A.n)
end

Base.:*(α::Real, A::SkewSymMatrix) = A*α

function Base.zeros(ST::Type{SkewSymMatrix{<:Real}}, n::Int)
    zeros(CPU(), ST, n)
end

function Base.zeros(backend::KernelAbstractions.Backend, ::Type{SkewSymMatrix{T}}, n::Int) where T
    zero_vec = if n != 1
        KernelAbstractions.zeros(backend, T, n*(n-1)÷2)
    else
        KernelAbstractions.allocate(backend, T, n*(n-1)÷2)
    end
	SkewSymMatrix(zero_vec, n)
end

function Base.zeros(::Type{SkewSymMatrix}, n::Int)
    SkewSymMatrix(zeros(n*(n-1)÷2), n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SkewSymMatrix{T}}, n::Int) where T
    SkewSymMatrix(rand(rng, T, n*(n-1)÷2),n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SkewSymMatrix}, n::Int)
    SkewSymMatrix(rand(rng, n*(n-1)÷2), n)
end

# TODO: make defaults when no rng is specified!!! (prbabaly rng ← Random.default_rng())
function Base.rand(type::Type{SkewSymMatrix{T}}, n::Integer) where T
    rand(Random.default_rng(), type, n)
end

function Base.rand(type::Type{SkewSymMatrix}, n::Integer)
    rand(Random.default_rng(), type, n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend, type::Type{SkewSymMatrix{T}}, n::Integer) where T 
    S = KernelAbstractions.allocate(backend, T, n*(n-1)÷2)
    Random.rand!(rng, S)
    SkewSymMatrix(S, n)
end

function Base.rand(backend::KernelAbstractions.Backend, type::Type{SkewSymMatrix{T}}, n::Integer) where T 
    rand(Random.default_rng(), backend, type, n)
end

#these are Adam operations:
function scalar_add(A::SkewSymMatrix, δ::Real)
    SkewSymMatrix(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::SkewSymMatrix)
    SkewSymMatrix(A.S.^2, A.n)
end
function racᵉˡᵉ(A::SkewSymMatrix)
    SkewSymMatrix(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n 
    SkewSymMatrix(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::SkewSymMatrix, A::SkewSymMatrix, α::Real)
    mul!(C.S, A.S, α)
end
LinearAlgebra.mul!(C::SkewSymMatrix, α::Real, A::SkewSymMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::SkewSymMatrix, α::Real) = mul!(C, C, α)

function Base.:*(A::SkewSymMatrix{T}, B::AbstractMatrix{T}) where T
    m1, m2 = size(B)
    @assert m1 == A.n
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.allocate(backend, T, A.n, m2)

    skew_mat_mul! = skew_mat_mul_kernel!(backend)
    skew_mat_mul!(C, A.S, B, A.n, ndrange=size(C))
    C
end

@kernel function skew_mat_mul_kernel!(C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where T
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = 1:(i-1)
        tmp_sum +=  S[(i-2)*(i-1)÷2+k] * B[k, j]
    end
    for k = (i+1):n 
        tmp_sum += -S[(k-2)*(k-1)÷2+i] * B[k, j]
    end
    C[i,j] = tmp_sum
end

function Base.:*(B::AbstractMatrix{T}, A::SkewSymMatrix{T}) where T 
    (-A*B')'
end

function Base.:*(A::SkewSymMatrix, b::AbstractVector{T}) where T
    A*reshape(b, length(b), 1)
end

function Base.one(A::SkewSymMatrix{T}) where T
    backend = KernelAbstractions.get_backend(A.S)
    unit_matrix = KernelAbstractions.zeros(backend, T, A.n, A.n)
    write_ones! = write_ones_kernel!(backend)
    write_ones!(unit_matrix, ndrange=A.n)
    unit_matrix
end


# the first matrix is multiplied onto A2 in order for it to not be SkewSymMatrix!
function Base.:*(A1::SkewSymMatrix{T}, A2::SkewSymMatrix{T}) where T 
    A1 * (one(A2) * A2) 
end

@doc raw"""
    vec(A)

Output the associated vector of `A`.

# Examples

```jldoctest
using GeometricMachineLearning

M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
SkewSymMatrix(M) |> vec

# output

6-element Vector{Float64}:
 1.5
 3.0
 1.5
 4.5
 3.0
 1.5
```
"""
function Base.vec(A::SkewSymMatrix)
    A.S
end

function Base.zero(A::SkewSymMatrix)
    SkewSymMatrix(zero(A.S), A.n)
end

function KernelAbstractions.get_backend(A::SkewSymMatrix)
    KernelAbstractions.get_backend(A.S)
end

function assign!(B::SkewSymMatrix{T}, C::SkewSymMatrix{T}) where T 
    B.S .= C.S 
end

function Base.copy(A::SkewSymMatrix)
    SkewSymMatrix(copy(A.S), A.n)
end

@kernel function assign_Skew_val_kernel!(S, A_skew, i)
    j = @index(Global)
    S[((i - 2) * (i - 1) ÷ 2 + j)] = A_skew[i, j]
end

function map_to_Skew(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    @assert size(A, 2) == n
    A_skew = T(.5)*(A - A')
    backend = KernelAbstractions.get_backend(A)
    S = if n != 1
        KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    else
        KernelAbstractions.allocate(backend, T, n * (n - 1) ÷ 2)
    end
    assign_Skew_val! = assign_Skew_val_kernel!(backend)
    for i in 2:n
        assign_Skew_val!(S, A_skew, i, ndrange = (i - 1))
    end
    S
end

function map_to_Skew(A::AbstractMatrix{T}) where T <: Integer
    Float = T == Int64 ? Float64 : Float32
    map_to_Skew(Float.(A))
end

function Base.copyto!(A::SkewSymMatrix, B::SkewSymMatrix)
    A.S .= B.S
    nothing
end

function _round(A::SkewSymMatrix; kwargs...)
    SkewSymMatrix(_round(A.S; kwargs...), A.n)
end

function _round(A::AbstractArray; kwargs...)
    round.(A; kwargs...)
end

# define routines for generalizing ChainRulesCore to SkewSymMatrix 
ChainRulesCore.ProjectTo(A::SkewSymMatrix) = ProjectTo{SkewSymMatrix}(; skew_sym = ProjectTo(A.S))
(project::ProjectTo{SkewSymMatrix})(dA::AbstractMatrix) = SkewSymMatrix(project.skew_sym(map_to_Skew(dA)), size(dA, 2))
(project::ProjectTo{SkewSymMatrix})(dA::SkewSymMatrix) = SkewSymMatrix(project.skew_sym(dA.S), dA.n)