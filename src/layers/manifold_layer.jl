# This defines a manifold layer that only has one matrix-valued manifold $A$ associated with it does $x\mapsto{}Ax$. 
abstract type ManifoldLayer{M, N} <: AbstractExplicitLayer{M, N} end

function (d::ManifoldLayer{M, N})(x::AbstractVecOrMat, ps::NamedTuple) where {M, N}
    N > M ? ps.weight*x : ps.weight'*x
end

function (d::ManifoldLayer{M, N})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T}
    N > M ? mat_tensor_mul(ps.weight, x) : mat_tensor_mul(ps.weight', x)
end
