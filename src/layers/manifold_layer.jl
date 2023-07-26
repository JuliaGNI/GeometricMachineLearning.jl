@doc raw"""
This defines a manifold layer that only has one matrix-valued manifold $A$ associated with it does $x\mapsto{}Ax$. 
"""
abstract type ManifoldLayer{N, M, reverse} <: AbstractExplicitLayer{N, M} end

function (d::ManifoldLayer{N, M, false})(x::AbstractArray, ps::NamedTuple) where {N, M}
    ps.weight*x
end

function (d::ManifoldLayer{N, M, true})(x::AbstractArray, ps::NamedTuple) where {N, M}
    ps.weight'*x
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