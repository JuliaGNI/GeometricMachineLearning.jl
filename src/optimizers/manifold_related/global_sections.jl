@doc raw"""
This implements global sections for the Stiefel manifold and the Symplectic Stiefel manifold. 

In practice this is implemented using Householder reflections, with the auxiliary column vectors given by: 
|0|
|0|
|.|
|1| ith spot for i in (n+1) to N (or with random columns)
|0|
|.|
|0|

Maybe consider dividing the output in the check functions by n!

Implement a general global section here!!!! Tₓ𝔐 → G×𝔤 !!!!!! (think about random initialization!)
"""
struct GlobalSection{T, AT} 
    Y::AT
    # for now the only lift that is implemented is the Stiefel one - these types will have to be expanded!
    λ::Union{LinearAlgebra.QRCompactWYQ, LinearAlgebra.QRPackedQ, Nothing}

    function GlobalSection(Y::AbstractVecOrMat)
        λ = global_section(Y)
       new{eltype(Y), typeof(Y)}(Y, λ) 
    end
end

function GlobalSection(ps::NamedTuple)
    apply_toNT(GlobalSection, ps)
end

# this is an application G×𝔐 → 𝔐
function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)
    @assert (N, n) == size(Y₂)
    backend = KernelAbstractions.get_backend(Y₂)
    StiefelManifold(
        λY.Y.A * Y₂.A[1:n, :] + λY.λ*vcat(Y₂.A[(n+1):N, :], KernelAbstractions.zeros(backend, T, n, n))
    )
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)
    @assert (N, n) == size(Y₂) == size(Y)

    backend = KernelAbstractions.get_backend(Y)
    Y.A .= λY.Y * Y₂.A[1:n, :] + λY.λ*vcat(Y₂.A[(n+1):N, :], KernelAbstractions.zeros(backend, T, n, n))
end

function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:GrassmannManifold{T}}
    N, n = size(λY.Y)
    @assert (N, n) == size(Y₂)
    GrassmannManifold(λY.λ*Y₂)
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:GrassmannManifold{T}}
    N, n = size(λY.Y)
    @assert (N, n) == size(Y₂)
    Y.A = λY.λ*Y₂
end

function apply_section(λY::GlobalSection{T}, Y₂::AbstractVecOrMat{T}) where {T}
    λY.Y + Y₂
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AbstractVecOrMat{T}) where {T, AT<:AbstractVecOrMat{T}}
    Y .= Y₂ + λY.Y
end

function apply_section(λY::NamedTuple, Y₂::NamedTuple)
    apply_toNT(apply_section, λY, Y₂)
end

function apply_section!(Y::NamedTuple, λY::NamedTuple, Y₂::NamedTuple)
    apply_toNT(apply_section!, Y, λY, Y₂)
end

function global_rep(λY::NamedTuple, gx::NamedTuple)
    apply_toNT(global_rep, λY, gx)
end

##auxiliary function 
function global_rep(::GlobalSection{T}, gx::AbstractVecOrMat{T}) where {T}
    gx
end

function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)
    StiefelLieAlgHorMatrix(
        SkewSymMatrix(λY.Y.A' * Δ),
        (λY.λ' * Δ)[1:(N-n), 1:n], 
        # (λY.λ' * Δ)[(n+1):N, 1:n],
        N, 
        n
    )
end

function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:GrassmannManifold{T}}
    N, n = size(λY.Y)
    GrassmannLieAlgHorMatrix(
        (λY.λ' * Δ)[(n+1):N, 1:n],
        N,
        n
    )
end