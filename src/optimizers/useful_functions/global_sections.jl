"""
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
#global section maps an element of the manifold to its associated Lie group!
struct GlobalSection{T, AT} 
    Y::AT
    #for now the only lift that is implemented is the Stiefel one - these types will have to be expanded!
    λ::Union{LinearAlgebra.QRCompactWYQ, LinearAlgebra.QRPackedQ, Nothing}

    function GlobalSection(Y::AbstractVecOrMat)
        λ = global_section(Y)
       new{eltype(Y), typeof(Y)}(Y, λ) 
    end
end

function GlobalSection(ps::NamedTuple)
    apply_toNT(ps, GlobalSection)
end

#this is an application G×𝔐 → 𝔐
function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)
    @assert (N, n) == size(Y₂)
    StiefelManifold(
        λY.Y*Y₂[1:n,1:n] + λY.λ*vcat(Y₂[n+1:N,1:n], zeros(n, n))
    )
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)
    @assert (N, n) == size(Y₂) == size(Y)
    Y.A .= λY.Y*Y₂[1:n,1:n] + λY.λ*vcat(Y₂[n+1:N,1:n], zeros(n, n))
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

function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:AbstractVecOrMat{T}}
    Y .= Y₂ + λY.Y
end

function apply_section(λY::NamedTuple, Y₂::NamedTuple)
    apply_toNT(λY, Y₂, apply_section)
end

function apply_section!(Y::NamedTuple, λY::NamedTuple, Y₂::NamedTuple)
    apply_toNT(Y, λY, Y₂, apply_section!)
end

function global_rep(λY::NamedTuple, gx::NamedTuple)
    apply_toNT(λY, gx, global_rep)
end

##auxiliary function 
function global_rep(::GlobalSection{T}, gx::AbstractVecOrMat{T}) where {T}
    gx
end

function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)
    StiefelLieAlgHorMatrix(
        SkewSymMatrix(λY.Y'*Δ),
        (λY.λ'*Δ)[1:N-n,1:n], 
        N, 
        n
    )
end

function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:GrassmannManifold{T}}
    N, n = size(λY.Y)
    GrassmannLieAlgHorMatrix(
        (λY.λ'*Δ)[n+1:N,1:n],
        N,
        n
    )
end

#I might actually not need this!
function Ω(U::SymplecticStiefelManifold{T}, Δ::AbstractMatrix{T}) where {T} 
    J_mat = SymplecticPotential(T, size(U,1)÷2)
    SymplecticLieAlgMatrix(
        Δ*inv(U'*U)*U' + J_mat*U*inv(U'U)*Δ'*(I + J_mat*U*inv(U'*U)*U'*J_mat)*J_mat
    )
end

Ω₁(Y::StiefelManifold, Δ::AbstractMatrix) = SkewSymMatrix(2*(I - .5*Y*Y')*Δ*Y') 
#TODO: perform calculations in-place, don't allocate so much!
function Ω(Y::StiefelManifold, Δ::AbstractMatrix)
    N = size(Y,1)
    B̃ = zeros(N, N)
    mul!(B̃, Δ, Y')
    B̂ = zero(B̃)
    mul!(B̂, Y, Y')
    rmul!(B̂, -.5)
    @views B̂ .+= one(B̂)
    B = zero(B̂)
    mul!(B, B̂, B̃)
    SkewSymMatrix(B)
end