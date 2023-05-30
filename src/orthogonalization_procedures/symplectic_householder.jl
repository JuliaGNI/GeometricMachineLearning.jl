"""
this algorithm is taken (and adjusted) from https://doi.org/10.1016/j.laa.2008.02.029
"""
struct Sympl_Householder_Decom{T, AT, VT} <: LinearAlgebra.Factorization where {AT<:AbstractMatrix, VT<:AbstractVector}
    A::AT
    c₁::VT
    c₂::VT 
    ρ::VT
    ν::VT
    μ::VT
    function Sympl_Householder_Decom(A::AbsractMatrix{T}, c₁::AbstractVector{T}, 
        c₂::AbstractVector{T}, ρ::AbstractVector{T}, ν::AbstractVector{T}, μ::AbstractVector{T})
        new{T, typeof(A), typeof(c₁)}(A, c₁, c₂, ρ, ν, μ)
    end
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
        @views b = A[row_ind, N+i]
        c₁[i], c₂[i], ρ[i], ν[i], μ[i] = symplectic_householder!(a, b)
        #apply Householder to rest of matrix
        if i < M
            col_ind = vcat((i+1):M, (M+i+1):M2)
            @views ̃A = A[row_ind, col_ind]
            ̃A += c₁[i]*J
        end
        for j in (i+1):M
            row_ind = vcat(i:N, (N+i):N2)
            A[row_ind, j] .+= c₁[i]*J(a, A[row_ind, i])*a
            A[row_ind, j] .+= c₂[i]*J(b, A[row_ind, i])*b
        end
    end
    Sympl_Householder_Decom(A, c₁, c₂, ρ, ν, μ)
end

function sqr(A::AbstractMatrix{T}) where {T}
    A_copy = copy(A)
    sqr!(A_copy)
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

function symplectic_householder_step!(A::AbstractMatrix)
    N2, M2 = size(A)
    @assert iseven(M2)
    N = N2÷2
    M = M2÷2
    c₁, c₂, _, _, _, v₁, v₂ = symplectic_householder(A[:,1], A[:,M+1])
    for i in 1:M2
        A[:,i] .+= c₁*J(v₁, A[:,i])*v₁
        A[:,i] .+= c₂*J(v₂, A[:,i])*v₂
    end
end

function symplectic_householder_full!(A::AbstractMatrix)
    N2, M2 = size(A)
    @assert iseven(M2)
    @assert iseven(N2)
    N = N2÷2
    M = M2÷2
    for i in ((1:M) .+ 1)
        row_ind = vcat((i-1):N, (N+i-1):N2)
        col_ind = vcat((i-1):M, (M+i-1):M2)
        @views A_temp = A[row_ind, col_ind]
        symplectic_householder_step!(A_temp)
    end
end

