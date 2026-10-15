# This defines a manifold layer that only has one matrix-valued manifold $A$ associated with it does $x\mapsto{}Ax$. 
abstract type ManifoldLayer{M, N} <: AbstractExplicitLayer{M, N} end

function (d::ManifoldLayer{M, N})(x::AbstractArray, ps::NamedTuple) where {M, N}
    N > M ? _mul(ps.weight, x) : _mul(ps.weight', x)
end
