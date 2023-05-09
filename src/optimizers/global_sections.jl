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
struct GlobalSection{AT <: Lux.AbstractExplicitLayer}
    Y::NamedTuple
    λ::Union{NamedTuple, Nothing}

    function GlobalSection(d::Lux.AbstractExplicitLayer, ps::NamedTuple)
        new{typeof{d}}(ps, nothing)
    end

    function GlobalSection(d::ManifoldLayer, ps::NamedTuple)
       new{typeof(d)}(ps, global_section(ps)) 
    end
end

function global_section(d::ManifoldLayer, ps::NamedTuple)
    (weight = global_section(d, ps.weight), )
end

#function to improve readability when dealing with NamedTuple. 
function Base.:*(λ::NamedTuple{(:weight,), Tuple{AT}}, x::AbstractVecOrMat) where AT <: LinearAlgebra.QRCompactWYQ
    λ.weight*x
end

#one could include information about the orthogonalization procedure in the manifold layer (but this is householder most of the time)
function global_section(d::StiefelLayer, Y::StiefelManifold)
    N, n = size(Y)
    A = randn(N, N-n)
    A = A - Y*Y'*A
    qr(A).Q
end

#this is an application G×𝔐 → 𝔐
function apply(λY::GlobalSection{StiefelLayer}, Y₂::StiefelManifold)
    N, n = size(λY.Y.weight)
    StiefelManifold(
        λY.Y*Y₂[1:n,1:n] + λY.λ*vcat(Y₂[n+1:N,1:n], zeros(n, n))
    )
end
function apply(λY::GlobalSection{AT}, Y₂::NamedTuple{(:weight,), Tuple{BT}}) where {AT <: StiefelLayer, BT <: StiefelManifold}
    (weight = λY(Y₂.weight), )
end
#this is less secure!
#function apply(λY::GlobalSection{AT}, Y₂::NamedTuple{(:weight,), Tuple{BT}}) where {AT <: ManifoldLayer, BT <: Manifold}
#    (weight = λY(Y₂.weight), )
#end

function apply(λY::GlobalSection, ps₂::NamedTuple)
    for key in keys(λY.Y)
        λY.Y[key] += ps₂[key]
    end
end

#I might actually not need this!
Ω₁(Y::StiefelManifold, Δ::AbstractMatrix) = SkewSymMatrix((I - .5*Y*Y')*Δ*Y') 
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

function global_rep(λY::GlobalSection{StiefelLayer}, gx::NamedTuple)
    (weight = global_rep(λY, gx.weight), )
end

function global_rep(λY::GlobalSection, gx::NamedTuple)
    gx
end