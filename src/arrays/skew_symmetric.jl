@doc raw"""
A `SkewSymMatrix` is a matrix ``A`` s.t. ``A^T = -A``.

If the constructor is called with a matrix as input it returns a symmetric matrix via the projection ``A \mapsto \frac{1}{2}(A - A^T)``. 
This is a projection defined via the canonical metric ``\mathbb{R}^{n\times{}n}\times\mathbb{R}^{n\times{}n}\to\mathbb{R}, (A,B) \mapsto \mathrm{Tr}(A^TB)``.

The first index is the row index, the second one the column index.

The struct two fields: `S` and `n`. The first stores all the entries of the matrix in a sparse fashion (in a vector) and the second is the dimension ``n`` for ``A\in\mathbb{R}^{n\times{}n}``.
"""

mutable struct SkewSymMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int

    function SkewSymMatrix(S::AbstractVector{T},n::Int) where {T}
        @assert length(S) == n*(n-1)÷2
        new{T,typeof(S)}(S,n)
    end
    function SkewSymMatrix(S::AbstractMatrix{T}) where {T}
        n = size(S, 1)
        @assert size(S, 2) == n
        S_vec = zeros(T, n*(n-1)÷2)
        # make the input skew-symmetric if it isn't already
        S = T(.5)*(S - S')
        # this is disgusting and should be removed! Here because indexing for GPUs not supported.
        S_cpu = Matrix{T}(S)
        # map the sub-diagonal elements to a vector 
        for i in 2:n
            S_vec[((i-1)*(i-2)÷2+1):(i*(i-1)÷2)] = S_cpu[i,1:(i-1)]
        end
        S_vec₂ = Base.typename(typeof(S)).wrapper{eltype(S), 1}(S_vec)
        new{T,typeof(S_vec₂)}(S_vec₂, n)
    end
end 

function Base.getindex(A::SkewSymMatrix, i::Int, j::Int)
    if j == i
        return zero(eltype(A))
    end
    if i > j
        return A.S[(i-2)*(i-1)÷2+j]
    end
    return - A.S[(j-2)*(j-1)÷2+i]
end


Base.parent(A::SkewSymMatrix) = A.S
Base.size(A::SkewSymMatrix) = (A.n,A.n)

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

function Base.zeros(::Type{SkewSymMatrix{T}}, n::Int) where T
    SkewSymMatrix(zeros(T, n*(n-1)÷2), n)
end

function Base.zeros(backend::KernelAbstractions.Backend, ::Type{SkewSymMatrix{T}}, n::Int) where T
	SkewSymMatrix(KernelAbstractions.zeros(backend, T, n*(n-1)÷2), n)
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
    i,j = @index(Global, NTuple)

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
    A*reshape(b, size(b), 1)
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
If `vec` is applied onto `SkewSymMatrix`, then the output is the associated vector.  
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