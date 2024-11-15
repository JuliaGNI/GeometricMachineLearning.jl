"""
Routines for the symplectic Stiefel manifold.
"""

mutable struct SymplecticStiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
    function SymplecticStiefelManifold(A::AbstractMatrix)
        @assert iseven(size(A)[1])
        @assert iseven(size(A)[2])
        @assert size(A,1) ≥ size(A,2)
        new{eltype(A), typeof(A)}(A)
    end
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymplecticStiefelManifold{T}}, N2::Integer, n2::Integer) where {T}
    @assert N2 ≥ n2
    A = randn(rng, T, N2, n2)
    N, n = N2÷2, n2÷2
    SymplecticStiefelManifold(sr!(A).S[1:N2, vcat(1:n, (N+1):(N+n))])
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymplecticStiefelManifold}, N2::Integer, n2::Integer)
    @assert N2 ≥ n2
    A = randn(rng, N2, n2)
    N, n = N2÷2, n2÷2
    SymplecticStiefelManifold(sr!(A).S[1:N2, vcat(1:n, (N+1):(N+n))])
end

function Base.rand(::Type{SymplecticStiefelManifold{T}}, N2::Integer, n2::Integer) where {T}
    @assert N2 ≥ n2
    A = randn(T, N2, n2)
    N, n = N2÷2, n2÷2
    SymplecticStiefelManifold(sr!(A).S[1:N2, vcat(1:n, (N+1):(N+n))])
end

function Base.rand(::Type{SymplecticStiefelManifold}, N2::Integer, n2::Integer)
    @assert N2 ≥ n2
    A = randn(N2, n2)
    N, n = N2÷2, n2÷2
    SymplecticStiefelManifold(sr!(A).S[1:N2, vcat(1:n, (N+1):(N+n))])
end


function rgrad(U::SymplecticStiefelManifold, e_grad::AbstractMatrix, J::AbstractMatrix=PoissonTensor(eltype(U),size(U,1)÷2))
    e_grad * (U' * U) + J * U * (e_grad' * J * U)
end


#metric taken from arxiv.org/abs/2108.12447
function metric(U::SymplecticStiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    J_mat = PoissonTensor(size(U,1)÷2)
    LinearAlgebra.tr(inv(U'U)*Δ₁'*(I - .5*J_mat'*U*inv(U'U)*U'*J_mat)Δ₂)
end

function check(U::SymplecticStiefelManifold{T}) where {T}
    N = size(U,1)÷2
    n = size(U,2)÷2
    norm(U'*PoissonTensor(T, N)*U - PoissonTensor(T, n))
end


function global_section(U::SymplecticStiefelManifold)
    N2, n2 = size(U)
    A = randn(eltype(U), N2, N2-n2)
    J₁ = PoissonTensor(N2÷2)
    J₂ = PoissonTensor(n2÷2)
    A -= U*J₂*U'*J₁'*A
    sr!(A).S
end
