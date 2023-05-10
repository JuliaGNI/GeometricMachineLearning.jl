abstract type ManifoldLayer <: Lux.AbstractExplicitLayer end

#function to improve readability when dealing with NamedTuple (for Manifold layers)
function Base.:*(Y::NamedTuple{(:weight, ), Tuple{AT}}, x::AbstractVecOrMat) where AT <: Manifold
    Y.weight*x
end

function rgrad(d::ManifoldLayer, ps, dx)
    (weight = rgrad(ps.weight, dx.weight), )
end