@doc raw"""
This defines a manifold layer that only has one matrix-valued manifold $A$ associated with it does $x\mapsto{}Ax$. 
"""
abstract type ManifoldLayer{M, N, retraction} <: AbstractExplicitLayer{M, N} end

function (d::ManifoldLayer{M, N})(x::AbstractArray, ps::NamedTuple) where {M, N}
    N > M ? ps.weight*x : ps.weight'*x
end

function retraction(::ManifoldLayer{N, M, Geodesic}, B::NamedTuple{(:weight,),Tuple{AT}}) where {N, M, AT<:AbstractLieAlgHorMatrix}
    geodesic(B)
end

function retraction(::ManifoldLayer{N, M, Cayley}, B::NamedTuple{(:weight,),Tuple{AT}}) where {N, M, AT<:AbstractLieAlgHorMatrix}
    cayley(B)
end

#=
#function to improve readability when dealing with NamedTuple (for Manifold layers)
function Base.:*(Y::NamedTuple{(:weight, ), Tuple{AT}}, x::AbstractVecOrMat) where AT <: Manifold
    Y.weight*x
end

function rgrad(d::ManifoldLayer, ps, dx)
    (weight = rgrad(ps.weight, dx.weight), )
end
=#