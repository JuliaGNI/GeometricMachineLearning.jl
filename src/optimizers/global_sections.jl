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
struct GlobalSection{AT<:Lux.AbstractExplicitLayer, BT<:NamedTuple, CT<:Union{NamedTuple,Nothing}}
    Y::BT
    λ::CT

    function GlobalSection(d::Lux.AbstractExplicitLayer, ps::NamedTuple)
        new{typeof(d), typeof(ps), Nothing}(ps, nothing)
    end

    function GlobalSection(d::ManifoldLayer, ps::NamedTuple{(:weight,), Tuple{BT}}) where BT <: Manifold
        B = global_section(ps)
       new{typeof(d), typeof(ps), typeof(B)}(ps, B) 
    end
end

function global_section(d::ManifoldLayer, ps::NamedTuple{(:weight,), Tuple{BT}}) where BT <: Manifold
    (weight = global_section(d, ps.weight), )
end

#maybe define a mapping StiefelManifold ↦ StiefelLayer to make this safe!
function apply(λY::GlobalSection{AT}, Y₂::NamedTuple{(:weight,), Tuple{BT}}) where {AT <: ManifoldLayer, BT <: Manifold}
    (weight = apply(λY, Y₂.weight), )
end

function apply(λY::GlobalSection, ps₂::NamedTuple)
    for key in keys(λY.Y)
        λY.Y[key] += ps₂[key]
    end
end

function global_rep(::AT, λY::GlobalSection{AT}, gx::NamedTuple) where AT<:ManifoldLayer
    (weight = global_rep(λY, gx.weight), )
end


##auxiliary function 
function global_rep(::Lux.AbstractExplicitLayer, λY::GlobalSection, gx::NamedTuple)
    gx
end

###### the following are particular to the Stifel manifold, may be further generalized!!
#function to improve readability when dealing with NamedTuple:
function Base.:*(λ::NamedTuple{(:weight,), Tuple{AT}}, x::AbstractVecOrMat) where AT <: LinearAlgebra.QRCompactWYQ
    λ.weight*x
end

#this is an application G×𝔐 → 𝔐
function apply(λY::GlobalSection{StiefelLayer}, Y₂::StiefelManifold)
    N, n = size(λY.Y.weight)
    StiefelManifold(
        λY.Y*Y₂[1:n,1:n] + λY.λ*vcat(Y₂[n+1:N,1:n], zeros(n, n))
    )
end

function global_rep(λY::GlobalSection{StiefelLayer}, Δ::AbstractMatrix)
    N, n = size(λY.Y.weight)
    B = StiefelLieAlgHorMatrix(
        SkewSymMatrix(Y'*Δ),
        (λY.λ'*Δ)[1:N-n,1:n], 
        N, 
        n
    )
    B
end

#I might actually not need this!
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