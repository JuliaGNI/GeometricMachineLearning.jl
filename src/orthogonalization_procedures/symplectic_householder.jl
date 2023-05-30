N = 5
M = 4

A = rand(2*N, 2*M)

function compute_a2(A, ν=1)
    N, M = size(A).÷2
    if ν > M
        print("\nfuck!!!\n")
    end
    v₁ = copy(A[:,ν])
    v₂ = copy(A[:,M+ν])
    α = v₁[1:N]'*v₂[N+1:2*N] - v₁[N+1:2*N]'*v₂[1:N]
    a2 = α/(v₂[N+1]/v₁[1] - v₂[1]*v₁[N+1]/v₁[1]^2)
    if a2 < 0
        v₁, v₂, a2, ν =  compute_a2(A, ν+1)
    end
    return v₁, v₂, a2, ν
end

#rotate matrix into the format where two columns are zeroed out
function zero_rot!(A)
    N, M = size(A).÷2
    v₁, v₂, a2, ν = compute_a2(A)
    a = √(a2)
    b = v₂[1]/v₁[1]*a
    c = v₁[N+1]/v₁[1]*a
    d = v₂[N+1]/v₁[1]*a
    #factor by which the symplectic Householder has to be multiplied - implement this!
    #fac = 2*α - ...
    v₁[1] -= a 
    v₁[N+1] -= c 
    v₂[1] -= b
    v₂[N+1] -= d
    α = v₁[1:N]'*v₂[N+1:2*N] - v₁[N+1:2*N]'*v₂[1:N]
    #LinearAlgebra.rmul!(v₁, -1.)
    #LinearAlgebra.rmul!(v₂, -1.)
    V = hcat(v₁, v₂)
    VJ = hcat(vcat(v₂[N+1:2*N]', -v₁[N+1:2*N]'), vcat(-v₂[1:N]', v₁[1:N]'))
    A .-= 2/α*V*(VJ*A)
    ν, V
end

function factorize!(A, ν=(), V=())
    N, M = size(A).÷2
    print(N," ", M)
    if M > 0
        ν₁, V₁ = zero_rot!(A)
        index = vcat(1:(ν₁-1),(ν₁+1):M)
        index = vcat(index, index .+ M)
        @views A_new = A[vcat(2:N, N+2:2*N), index]
        ν, V =  factorize!(A_new, ν, V)
        return (ν₁, ν...), (V₁, V...)
    end
    ν, V    
end

"""
this algorithm is taken (and adjusted) from https://doi.org/10.1016/j.laa.2008.02.029
"""

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
    #μ = b[1] + ξ 
    s = +1.
    c₂ = -s /(ξ*ν)
    LinearAlgebra.rmul!(b, -1)
    b[1] =  s*ξ 
    b[N+1] = 0.
    c₁, c₂
end

function symplectic_householder(a::AbstractVector, b::AbstractVector)
    a_copy, b_copy = copy(a), copy(b)
    symplectic_householder!(a_copy, b_copy)..., a_copy, b_copy
end

function symplectic_householder!(A::AbstractMatrix)
    N2, M2 = size(A)
    @assert iseven(M2)
    N = N2÷2
    M = M2÷2
    c₁, c₂, v₁, v₂ = symplectic_householder(A[:,1], A[:,M+1])
    for i in 1:M2
        A[:,i] .+= c₁*J(v₁, A[:,i])*v₁
        A[:,i] .+= c₂*J(v₂, A[:,i])*v₂
    end
end
