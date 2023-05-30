"""
this algorithm is taken (and adjusted) from https://doi.org/10.1016/j.laa.2008.02.029
"""
struct SymplecticHouseholderDecom{T, AT<:AbstractMatrix{T}, VT<:AbstractVector{T}} <: LinearAlgebra.Factorization{T} #where {AT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    A::AT
    c₁::VT
    c₂::VT 
    ρ::VT
    ν::VT
    μ::VT
    function SymplecticHouseholderDecom(A::AbstractMatrix{T}, c₁::AbstractVector{T}, 
        c₂::AbstractVector{T}, ρ::AbstractVector{T}, ν::AbstractVector{T}, μ::AbstractVector{T}) where {T}
        new{T, typeof(A), typeof(c₁)}(A, c₁, c₂, ρ, ν, μ)
    end
end

struct Sfac{T, ST} <: AbstractMatrix{T} where {ST<:SymplecticHouseholderDecom{T}}
    Λ::ST
    function Sfac(Λ::SymplecticHouseholderDecom{T}) where T
        new{T, typeof(Λ)}(Λ)
    end
end

function Base.size(S::Sfac) 
    N2 = size(S.Λ.A, 1)
    (N2, N2)
end

#this is incredibly inefficient but may not be needed.
function Base.getindex(S::Sfac, i::Integer, j::Integer)
    S_mat = one(ones(size(S)...))
    apply_S!(S.Λ, S_mat)
    S_mat[i, j]
end

function Base.:*(S::Sfac, B::AbstractMatrix)
    apply_S(S.Λ, B)
end

function Base.:*(B::AbstractMatrix, S::Sfac)
end

function sqr!(A::AbstractMatrix{T}) where {T}
    N2, M2 = size(A)
    @assert iseven(N2)
    @assert iseven(M2)
    N = N2÷2
    M = M2÷2
    c₁ = zeros(T, M)
    c₂ = zeros(T, M)
    ρ = zeros(T, M)
    ν = zeros(T, M)
    μ = zeros(T, M)
    for i in 1:M
        row_ind = vcat(i:N, (N+i):N2)
        @views a = A[row_ind, i]
        @views b = A[row_ind, M+i]
        c₁[i], c₂[i], ρ[i], ν[i], μ[i] = symplectic_householder!(a, b)
        #apply Householder to rest of matrix
        for j in (i+1):M
            row_ind = vcat(i:N, (N+i):N2)
            A[row_ind, j] .+= c₁[i]*J(a, A[row_ind, i])*a
            A[row_ind, j] .+= c₂[i]*J(b, A[row_ind, i])*b
        end
    end
    SymplecticHouseholderDecom(A, c₁, c₂, ρ, ν, μ)
end

function sqr(A::AbstractMatrix{T}) where {T}
    A_copy = copy(A)
    sqr!(A_copy)
end

function apply_S!(Λ::SymplecticHouseholderDecom, B::AbstractMatrix)
    N, M = size(Λ.A).÷2
    for j in M:-1:1
        row_ind = vcat(j:N, (N+j):(2*N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j+M]
        for i in 1:size(B, 2)
            B[row_ind, i] .+= Λ.c₁[j]*J(v₁, B[row_ind, i])*v₁
            B[row_ind, i] .+= Λ.c₂[j]*J(v₂, B[row_ind, i])*v₂
        end
    end
    B
end

function apply_S(Λ::SymplecticHouseholderDecom, B::AbstractMatrix)
    B_copy = copy(B)
    apply_S!(Λ, B_copy)
    B_copy
end

function J(a::AbstractVector, b::AbstractVector)
    N2 = length(a)
    @assert iseven(N2)
    @assert length(b) == N2
    N = N2÷2
    -a[(N+1):2*N]'*b[1:N] + a[1:N]'*b[(N+1):2*N]
end


#you probably also have to return the values ρ, ν, μ
function symplectic_householder!(a::AbstractVector, b::AbstractVector)
    N2 = length(a)
    @assert iseven(N2)
    @assert length(b) == N2
    N = N2÷2
    ρ = sign(a[1])*LinearAlgebra.norm(a)
    c₁ = 1/(ρ*a[N+1])
    a[1] -= ρ
    LinearAlgebra.rmul!(a, -1)
    b .+= c₁*J(a,b)*a 
    ν = b[N+1]
    ξ = sqrt(LinearAlgebra.norm(b)^2 - b[1]^2 - ν^2)
    #look at the choice of pre-sign!
    s = +1.
    μ = b[1] + s*ξ
    c₂ = s /(ξ*ν)
    LinearAlgebra.rmul!(b, -1)
    b[1] =  s*ξ 
    b[N+1] = 0.
    c₁, c₂, ρ, ν, μ
end

function symplectic_householder(a::AbstractVector, b::AbstractVector)
    a_copy, b_copy = copy(a), copy(b)
    symplectic_householder!(a_copy, b_copy)..., a_copy, b_copy
end