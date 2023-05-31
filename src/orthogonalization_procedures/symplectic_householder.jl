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

struct Sfac{inverse, T, ST} <: AbstractMatrix{T} where {inverse, ST<:SymplecticHouseholderDecom{T}}
    Λ::ST
    function Sfac(Λ::SymplecticHouseholderDecom{T}, inverse::Bool=false) where T
        new{inverse, T, typeof(Λ)}(Λ)
    end
end

function Base.size(S::Sfac) 
    N2 = size(S.Λ.A, 1)
    (N2, N2)
end

#this is incredibly inefficient but may not be needed.
function Base.getindex(S::Sfac{false}, i::Integer, j::Integer)
    S_mat = one(ones(size(S)...))
    apply_S_left!(S_mat, S.Λ)
    S_mat[i, j]
end
function Base.getindex(S::Sfac{true}, i::Integer, j::Integer)
    S_mat = one(ones(size(S)...))
    apply_S_inverse_left!(S_mat, S.Λ)
    S_mat[i, j]
end

function Base.inv(S::Sfac{false})
    Sfac(S.Λ, true)
end
function Base.inv(S::Sfac{true})
    Sfac(S.Λ, false)
end

struct Rfac{T, ST} <: AbstractMatrix{T} where {ST<:SymplecticHouseholderDecom{T}}
    Λ::ST
    function Rfac(Λ::SymplecticHouseholderDecom{T}) where T
        new{T, typeof(Λ)}(Λ)
    end
end

function Base.size(R::Rfac)
    size(R.Λ.A)
end

function Base.getindex(R::Rfac, i::Integer, j::Integer)
    N, M = size(R.Λ.A).÷2
    if j ≤ M
        if i == j 
            return R.Λ.ρ[i]
        end
        if i < j 
            return R.Λ.A[i, j]
        end
        if i ≤ N
            return 0.
        end
        if i < (N+j)
            return R.Λ.A[i, j]
        end
        return 0.
    end
    if i == (j-M)
        return R.Λ.μ[i]
    end
    if i == (j-M+N)
        return R.Λ.ν[i-N]
    end
    if i < (j-M)
        return R.Λ.A[i,j]
    end
    if i ≤ N
        return 0.
    end
    if i < (j-M+N) 
        return R.Λ.A[i,j]
    end
    return 0.
end

function Base.:*(S::Sfac{false}, B::AbstractMatrix)
    apply_S_left(S.Λ, B)
end

function Base.:*(S::Sfac{false}, b::AbstractVector)
    apply_S_left(S.Λ, b)
end

function Base.:*(S::Sfac{true}, B::AbstractMatrix)
    apply_S_inverse_left(S.Λ, B)
end

function Base.:*(S::Sfac{true}, B::AbstractVector)
    apply_S_inverse_left(S.Λ, b)
end

function Base.:*(B::AbstractMatrix, S::Sfac{false})
    apply_S_right(B, S.Λ)
end

struct SR{T, ST, RT} <: LinearAlgebra.Factorization{T} where {inverse, ST<:Sfac{inverse, T}, RT<:Rfac{T}}
    S::ST
    R::RT
    function SR(S::Sfac{inverse, T}, R::Rfac{T}) where {inverse, T}
        new{T, typeof(S), typeof(R)}(S, R)
    end
end

function sr!(A::AbstractMatrix{T}) where {T}
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
            A[row_ind, j] .+= c₁[i]*J(a, A[row_ind, j])*a
            A[row_ind, j] .+= c₂[i]*J(b, A[row_ind, j])*b
            A[row_ind, j+M] .+= c₁[i]*J(a, A[row_ind, j+M])*a
            A[row_ind, j+M] .+= c₂[i]*J(b, A[row_ind, j+M])*b
        end
    end
    Λ = SymplecticHouseholderDecom(A, c₁, c₂, ρ, ν, μ)
    SR(Sfac(Λ), Rfac(Λ))
end

function sr(A::AbstractMatrix{T}) where {T}
    A_copy = copy(A)
    sr!(A_copy)
end

function apply_S_left!(b::AbstractVector, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A).÷2
    for j in M:-1:1
        row_ind = vcat(j:N, (N+j):(2*N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j+M]
        b[row_ind] .-= Λ.c₂[j]*J(v₂, b[row_ind])*v₂
        b[row_ind] .-= Λ.c₁[j]*J(v₁, b[row_ind])*v₁
    end
    B
end

function apply_S_left!(B::AbstractMatrix, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A).÷2
    for j in M:-1:1
        row_ind = vcat(j:N, (N+j):(2*N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j+M]
        for i in 1:size(B, 2)
            B[row_ind, i] .-= Λ.c₂[j]*J(v₂, B[row_ind, i])*v₂
            B[row_ind, i] .-= Λ.c₁[j]*J(v₁, B[row_ind, i])*v₁
        end
    end
    B
end

function apply_S_inverse_left!(b::AbstractVector, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A).÷2
    for j in 1:M
        row_ind = vcat(j:N, (N+j):(2*N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j+M]
        b[row_ind] .+= Λ.c₁[j]*J(v₁, b[row_ind])*v₁
        b[row_ind] .+= Λ.c₂[j]*J(v₂, b[row_ind])*v₂
    end
    B
end

function apply_S_inverse_left!(B::AbstractMatrix, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A).÷2
    for j in 1:M
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

function apply_S_right!(B::AbstractMatrix, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A).÷2
    @assert size(B, 2) == 2*N
    for j in 1:M
        row_ind = vcat(j:N, (N+j):(2*N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j+M]
        for i in 1:size(B, 1)
            @views b = B[i, row_ind]
            fac₁ = -Λ.c₁[j]*b'*v₁
            b[1:(N+1-j)] .-= fac₁*v₁[(N+2-j):(2*N+2-2*j)]
            b[(N+2-j):(2*N+2-2*j)] .+= fac₁*v₁[1:(N+1-j)]
            fac₂ = -Λ.c₂[j]*b'*v₂
            b[1:(N+1-j)] .-= fac₂*v₂[(N+2-j):(2*N+2-2*j)]
            b[(N+2-j):(2*N+2-2*j)] .+= fac₂*v₂[1:(N+1-j)]
        end
    end
    B
end

function apply_S_left(Λ::SymplecticHouseholderDecom, B::AbstractArray)
    B_copy = copy(B)
    apply_S_left!(B_copy, Λ)
    B_copy
end

function apply_S_inverse_left(Λ::SymplecticHouseholderDecom, B::AbstractArray)
    B_copy = copy(B)
    apply_S_inverse_left!(B_copy, Λ)
    B_copy
end

function apply_S_right(B::AbstractArray, Λ::SymplecticHouseholderDecom)
    B_copy = copy(B)
    apply_S_right!(B_copy, Λ)
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