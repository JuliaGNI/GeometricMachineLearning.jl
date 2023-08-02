"""
maybe consider dividing the output in the check functions by n!
TODO: Implement sampling procedures!!
"""

mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
    function StiefelManifold(A::AbstractMatrix)
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
end

@kernel function assign_columns_kernel!(Y::AbstractMatrix{T}, A::AbstractMatrix{T}) where T
    i,j = @index(Global, NTuple)
    Y[i,j] = A[i,j]
end
function assign_columns(A::AbstractMatrix{T}, N::Integer, n::Integer) where T
    backend = KernelAbstractions.get_backend(A)
    Y = KernelAbstractions.allocate(backend, T, N, n)
    assign_columns! = assign_columns_kernel!(backend)
    assign_columns!(Y, A, ndrange=size(Y))
    Y
end

#TODO: check the distribution this is coming from - related to the Haar measure ???
function Base.rand(rng::Random.AbstractRNG, ::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T
    @assert N ≥ n
    A = randn(rng, T, N, n)
    StiefelManifold(assign_columns(qr!(A).Q, N, n))
end

function Base.rand(rng::Random.AbstractRNG, ::Type{StiefelManifold}, N::Integer, n::Integer)
    rand(rng, StiefelManifold{Float64}, N, n)
end

function Base.rand(manifold_type::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T
    rand(Random.default_rng(), manifold_type, N, n)
end

function Base.rand(::Type{StiefelManifold}, N::Integer, n::Integer)
    rand(StiefelManifold{Float64}, N, n)
end

function Base.rand(backend::KernelAbstractions.Backend, rng::Random.AbstractRNG, ::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T 
    @assert N ≥ n 
    A = KernelAbstractions.allocate(backend, T, N, n)
    Random.rand!(rng, A)
    StiefelManifold(qr!(A).Q[1:N, 1:n])
end

function Base.rand(backend::KernelAbstractions.Backend, manifold_type::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T 
    rand(backend, Random.default_rng(), manifold_type, N, n)
end

Base.:*(Y::StiefelManifold, B::AbstractMatrix) = Y.A*B
Base.:*(B::AbstractMatrix, Y::StiefelManifold) = B*Y.A
#this is needed for the implementation of MultiHeadAttention
function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::AbstractMatrix) where {T, AT<:AbstractGPUMatrix{T}}
    Y.parent.A'*B 
end

function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::AbstractMatrix) where {T, AT<:AbstractMatrix{T}}
    Y.parent.A'*B 
end

#function Base.rand(::TrivialInitRNG, ::Type{StiefelManifold{T}}, N::Int, n::Int) where T
#@assert N ≥ n 
#    zeros(StiefelLieAlgHorMatrix{T}, N, n)
#end

function Base.rand(::TrivialInitRNG{T}, ::Type{StiefelManifold}, N::Int, n::Int) where {T<:AbstractFloat}
    @assert N ≥ n 
    zeros(StiefelLieAlgHorMatrix{T}, N, n)
end

function rgrad(Y::StiefelManifold, e_grad::AbstractMatrix)
    e_grad - Y.A*(e_grad'*Y.A)
end

function metric(Y::StiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    LinearAlgebra.tr(Δ₁'*(I - .5*Y.A*Y.A')*Δ₂)
end

function check(Y::StiefelManifold)
    norm(Y.A'*Y.A - I)
end

function global_section(Y::StiefelManifold)
    N, n = size(Y)
    A = typeof(Y.A)(randn(eltype(Y), N, N-n))
    A = A - Y.A*Y.A'*A
    qr!(A).Q
end

function convert_to_dev(dev::Device, Y::StiefelManifold)
    StiefelManifold(convert_to_dev(dev, Y.A))
end

#same problem again 
function convert_to_dev(dev::CUDA.CuDevice, Y::StiefelManifold)
    StiefelManifold(convert_to_dev(dev, Y.A))
end
function convert_to_dev(dev::CPUDevice, Y::StiefelManifold)
    StiefelManifold(convert_to_dev(dev, Y.A))
end