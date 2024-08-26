@doc raw"""
    AbstractTriangular

See [`UpperTriangular`](@ref) and [`LowerTriangular`](@ref).
"""
abstract type AbstractTriangular{T} <: AbstractMatrix{T} end 

Base.parent(A::AbstractTriangular) = A.S
Base.size(A::AbstractTriangular) = (A.n, A.n)

function Base.:+(A::AT, B::AT) where AT <: AbstractTriangular
    @assert A.n == B.n 
    AT(A.S + B.S, A.n) 
end 

function add!(C::AT, A::AT, B::AT) where AT <: AbstractTriangular
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function Base.:-(A::AT, B::AT) where AT <: AbstractTriangular
    @assert A.n == B.n 
    AT(A.S - B.S, A.n) 
end 

function Base.:-(A::AT) where AT <: AbstractTriangular
    AT(-A.S, A.n)
end

function Base.:*(A::AT, α::Real) where AT <: AbstractTriangular
    AT(α * A.S, A.n)
end

Base.:*(α::Real, A::AT) where AT <: AbstractTriangular = A * α

function Base.zeros(backend::KernelAbstractions.Backend, ::Type{AT}, n::Int) where {T, AT <: AbstractTriangular{T}}
    # nameof converts AT to :UpperTriangular or ::LowerTriangular
	eval(nameof(AT))(KernelAbstractions.zeros(backend, T, n*(n-1)÷2), n)
end

function Base.zeros(::Type{AT}, n::Int) where {T, AT <: AbstractTriangular{T}}
    zeros(CPU(), AT, n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend, ::Type{AT}, n::Integer) where {T, AT <: AbstractTriangular{T}} 
    S = KernelAbstractions.allocate(backend, T, n*(n-1)÷2)
    Random.rand!(rng, S)
    eval(nameof(AT))(S, n)
end

function Base.rand(rng::Random.AbstractRNG, type::Type{AT}, n::Int) where {T, AT <: AbstractTriangular{T}}
    rand(rng, CPU(), type, n)
end

function Base.rand(type::Type{AT}, n::Integer) where {T, AT <: AbstractTriangular{T}}
    rand(Random.default_rng(), type, n)
end

function Base.rand(::Type{AT}, n::Integer) where {AT <: AbstractTriangular}
    rand(AT{Float64}, n)
end

function Base.rand(backend::KernelAbstractions.Backend, type::Type{AT}, n::Integer) where {T, AT <: AbstractTriangular{T}} 
    rand(Random.default_rng(), backend, type, n)
end

# these are Adam operations:
function scalar_add(A::AT, δ::Real) where {T, AT <: AbstractTriangular{T}}
    AT(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::AT) where AT <: AbstractTriangular
    AT(A.S.^2, A.n)
end
function racᵉˡᵉ(A::AT) where AT <: AbstractTriangular
    AT(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::AT, B::AT) where AT <: AbstractTriangular
    @assert A.n == B.n 
    AT(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::AT, A::AT, α::Real) where AT <: AbstractTriangular
    mul!(C.S, A.S, α)
end
LinearAlgebra.mul!(C::AT, α::Real, A::AT) where AT <: AbstractTriangular = mul!(C, A, α)
LinearAlgebra.rmul!(C::AT, α::Real) where AT <: AbstractTriangular = mul!(C, C, α)

function Base.one(A::AbstractTriangular{T}) where T
    backend = KernelAbstractions.get_backend(A.S)
    unit_matrix = KernelAbstractions.zeros(backend, T, A.n, A.n)
    write_ones! = write_ones_kernel!(backend)
    write_ones!(unit_matrix, ndrange=A.n)
    unit_matrix
end

# the first matrix is multiplied onto A2 in order for it to not be SkewSymMatrix!
function Base.:*(A1::AbstractTriangular{T}, A2::AbstractTriangular{T}) where T 
    A1 * (A2 * one(A2)) 
end

@doc raw"""
    vec(A::AbstractTriangular)

Return the associated vector to ``A``.

# Examples

```jldoctest
using GeometricMachineLearning

M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
LowerTriangular(M) |> vec

# output

6-element Vector{Int64}:
  5
  9
 10
 13
 14
 15
```
"""
function Base.vec(A::AbstractTriangular)
    A.S
end

function Base.zero(A::AT) where AT <: AbstractTriangular
    AT(zero(A.S), A.n)
end

function KernelAbstractions.get_backend(A::AbstractTriangular)
    KernelAbstractions.get_backend(A.S)
end

function assign!(B::AT, C::AT) where AT <: AbstractTriangular 
    B.S .= C.S 
end

function Base.copy(A::AT) where AT <: AbstractTriangular
    AT(copy(A.S), A.n)
end

function Base.copyto!(A::AbstractTriangular, B::AbstractTriangular)
    A.S .= B.S
    nothing
end

function Base.:*(A::AbstractTriangular, b::AbstractVector{T}) where T
    A * reshape(b, length(b), 1)
end

function Base.:*(B::AbstractMatrix{T}, A::AbstractTriangular{T}) where T 
    (A' * B')'
end