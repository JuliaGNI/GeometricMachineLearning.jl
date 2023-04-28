"""
Implementation of the regular Householder and the symplectic Householder reflection.

The multiplications here are different from what is normally done with Householder transforms and specific for our optimizers!

TODO: look at making this more efficient memory-wise!!!!
"""

#struct for householder decompositions
struct HouseDecom{transpose, T, AT <: AbstractMatrix{T}}
    QR::AbstractMatrix{T}
    τ::AbstractVector{T}

    function HouseDecom(A::AbstractMatrix)
        new{false, eltype(A), typeof(A)}(householder(A)...)
    end

    function Base.adjoint(HD::HouseDecom)
        new{true, eltype(HD.QR), typeof(HD.QR)}(HD.QR, HD.τ)
    end
end

#Base.'(HD::HouseDecom)

#NOTE: this is O(3) instead of O(2) -> should only be used for testing purposes!
function householderQ(A::AbstractMatrix)
    n, m = size(A)
    @assert n ≥ m
    Q = zeros(n,n)
    for i in 1:n Q[i,i] = 1. end 
    for i in 1:m 
        v = A[i:n, i]
        v = v/norm(v)
        v1 = v[1]
        e1 = zeros(n+1-i)
        e1[1] = 1.
        u = (v + sign(v1)*e1)
        u = u/norm(u)
        H = I(n+1-i) - 2*u*u'
        A[i:n,1:m] = H*A[i:n,1:m] 
        Q[1:n,i:n] = Q[1:n,i:n]*H
    end
    Q[1:n,1:m]
end

#Algorithm taken from https://www.ams.org/notices/200705/fea-mezzadri-web.pdf ... with modifications!
function householder!(A::AbstractMatrix)
    n, m = size(A)
    τ = zeros(m)
    for i = 1:m 
        norm_v = norm(A[i:n,i])
        s = sign(A[i,i])
        u₁ = A[i,i] + s*norm_v
        w = A[i:n,i]/u₁
        w[1] = 1
        A[i+1:n,i] = w[2:end]
        A[i,i] = -s*norm_v
        τ[i] = s*u₁/norm_v
        A[i:n, i+1:m] = A[i:n, i+1:m] - (τ[i]*w)*(w'*A[i:n, i+1:m])
    end
    τ
end 

function householder(A::AbstractMatrix)
    B = copy(A)
    τ = householder!(B)
    (B, τ)
end

#This defines QX (where Q is not square!!!)
function (HD::HouseDecom{false})(X)
    n, m = size(HD.QR)
    n2, m2 = size(X)
    @assert m == n2
    X_out = vcat(X, zeros(n-m, m2))
    for i = m:-1:1
        w = vcat(1., HD.QR[i+1:n,i])
        X_out[i:n, :] = X_out[i:n, :] - (HD.τ[i]*w)*(w'*X_out[i:n, :])
    end
    X_out
end

#Use adjoint from linear algebra to implement Q'X !!
function (HD::HouseDecom{true})(X)
    n, m = size(HD.QR)
    n2, m2 = size(X)
    @assert n == n2
    X_out = copy(X)
    for i = 1:m 
        w = vcat(1., HD.QR[i+1:end,i])
        X_out[i:n,:] = X_out[i:n,:] - (HD.τ[i]*w)*(w'*X_out[i:n,:])
    end
    X_out[1:m, :]
end 