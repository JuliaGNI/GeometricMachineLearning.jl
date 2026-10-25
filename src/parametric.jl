# The forward pass of a network that takes the parameters of the *system* — not of the network —
# beside its input. The system parameters `μ` are an `OptionalParameters`: a `NamedTuple`, or
# `NullParameters()` for a system without any.

# The flat vector of the system parameters, in the element type `T` where it is given, together with
# the `ParameterLayout` that puts such a vector back into the shape of `μ`. The layout is a value and
# not a closure, so a layer can store it in a field. `NullParameters` flattens to an empty vector.
_flatten_system_parameters(μ::NamedTuple) = flatten(μ)
_flatten_system_parameters(::NullParameters) = flatten(NamedTuple())
_flatten_system_parameters(::Type{T}, μ::NamedTuple) where {T} = flatten(T, μ)
_flatten_system_parameters(::Type{T}, ::NullParameters) where {T} = flatten(T, NamedTuple())

# The input of a parametric layer: the state with the flattened system parameters below it, in the
# element type of the state. One `μ` goes below every column of a matrix; a vector of them goes one
# below each column, so the columns are concatenated side by side. Without system parameters the state
# is the input as it is.
concatenate_array_with_parameters(x::AbstractArray, ::NullParameters) = x

function concatenate_array_with_parameters(x::AbstractVector, μ::NamedTuple)
    vcat(x, first(_flatten_system_parameters(eltype(x), μ)))
end

function concatenate_array_with_parameters(x::AbstractMatrix, μ::NamedTuple)
    flat = first(_flatten_system_parameters(eltype(x), μ))
    vcat(x, repeat(flat, 1, size(x, 2)))
end

function concatenate_array_with_parameters(x::AbstractMatrix, μs::AbstractVector)
    size(x, 2) == length(μs) || throw(DimensionMismatch(
        "the matrix has $(size(x, 2)) columns and there are $(length(μs)) system parameters"))
    reduce(hcat,
        [concatenate_array_with_parameters(x[:, i], μ) for (i, μ) in enumerate(μs)])
end

@doc raw"""
    apply_parametric(model, x, μ, ps)

Apply the `Chain` `model` with the network parameters `ps` to the input `x` and the parameters `μ`
of the system.

`x` is one sample — a vector, or a `(q, p)` named tuple of vectors — with one `μ`, an
`OptionalParameters`. Or `x` is a batch of `k` samples — a `2d × 1 × k` tensor, or a `(q, p)` named
tuple of `d × 1 × k` tensors — with a vector of `k` values of `μ`, one per sample, as a
[`ParametricDataLoader`](@ref) draws them. The output has the shape of the input.

The `Chain` receives the pair `(x, μ)`, so its first layer takes one sample and its `μ`, and a layer
that returns such a pair passes `μ` on to the next. A batch is applied sample by sample.

# Examples

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: apply_parametric
import AbstractNeuralNetworks: AbstractExplicitLayer

# a layer that adds the system parameter `a` to its input
struct AddParameter <: AbstractExplicitLayer{2, 2} end
(::AddParameter)((x, μ)::Tuple, ps) = x .+ μ.a

model = Chain(AddParameter())
ps = NetworkParameters((L1 = NamedTuple(),))
x = reshape([1.0, 2.0, 3.0, 4.0], 2, 1, 2)
apply_parametric(model, x, [(a = 10.0,), (a = 20.0,)], ps)

# output

2×1×2 Array{Float64, 3}:
[:, :, 1] =
 11.0
 12.0

[:, :, 2] =
 23.0
 24.0
```
"""
function apply_parametric(model::Chain,
        x::Union{AbstractVector, NamedTupleOfArrays{<:Any, (:q, :p)}},
        μ::OptionalParameters, ps::NetworkParameters)
    model((x, μ), ps)
end

function apply_parametric(model::Chain, x::AbstractArray{<:Any, 3}, μs::AbstractVector,
        ps::NetworkParameters)
    check_parametric_batch(size(x), μs)
    outputs = [apply_parametric(model, x[:, 1, i], μ, ps) for (i, μ) in enumerate(μs)]
    reshape(reduce(hcat, outputs), :, 1, length(μs))
end

function apply_parametric(model::Chain, x::NamedTupleOfArrays{<:Any, (:q, :p)},
        μs::AbstractVector, ps::NetworkParameters)
    check_parametric_batch(size(x.q), μs)
    outputs = [apply_parametric(model, (q = x.q[:, 1, i], p = x.p[:, 1, i]), μ, ps)
               for (i, μ) in enumerate(μs)]
    (q = reshape(reduce(hcat, [output.q for output in outputs]), :, 1, length(μs)),
        p = reshape(reduce(hcat, [output.p for output in outputs]), :, 1, length(μs)))
end

# A batch holds one time step per sample and one `μ` per sample.
function check_parametric_batch(input_size::NTuple{3, Int}, μs::AbstractVector)
    input_size[2] == 1 && input_size[3] == length(μs) || throw(DimensionMismatch(
        "a batch for `apply_parametric` has one time step and one system parameter per " *
        "sample; the input has $(input_size[2]) time steps and $(input_size[3]) samples, " *
        "and there are $(length(μs)) system parameters"))
    nothing
end
