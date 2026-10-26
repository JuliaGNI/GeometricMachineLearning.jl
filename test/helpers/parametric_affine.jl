# A layer whose forward pass takes the parameters `μ` of the system beside the state:
# `(x, μ) ↦ W [x; flatten(μ)] + b`. It acts on one sample, a vector `x` or a `(q, p)` pair of vectors
# that stacks `q` above `p`, which is what `apply_parametric` hands the `Chain`. The tests build the
# expected value of this map by hand, with the flattened `μ` written out.
using GeometricMachineLearning: concatenate_array_with_parameters
using AbstractNeuralNetworks: AbstractExplicitLayer, Initializer, NamedTupleOfArrays
import AbstractNeuralNetworks: initialparameters, parameterlength
import KernelAbstractions
import Random

struct ParametricAffine{M} <: AbstractExplicitLayer{M, M}
    parameter_length::Int
end

function initialparameters(
        rng::Random.AbstractRNG, init::Initializer, l::ParametricAffine{M},
        backend::KernelAbstractions.Backend, ::Type{T}) where {M, T}
    W = KernelAbstractions.zeros(backend, T, M, M + l.parameter_length)
    b = KernelAbstractions.zeros(backend, T, M)
    init(rng, W)
    init(rng, b)
    (W = W, b = b)
end

parameterlength(l::ParametricAffine{M}) where {M} = M * (M + l.parameter_length) + M

(l::ParametricAffine)(input::Tuple, ps::NamedTuple) = affine(input..., ps)

affine(x::AbstractVector, μ, ps) = ps.W * concatenate_array_with_parameters(x, μ) + ps.b

function affine(x::NamedTupleOfArrays{<:Any, (:q, :p)}, μ, ps)
    y = affine(vcat(x.q, x.p), μ, ps)
    n = length(x.q)
    (q = y[1:n], p = y[(n + 1):end])
end
