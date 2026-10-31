# The forward pass of a network that takes the parameters of the *system* — not of the network —
# beside its input. The system parameters `μ` are an `OptionalParameters`: a `NamedTuple`, or
# `NullParameters()` for a system without any.

"""
    _flatten_system_parameters(μ::NamedTuple)
    _flatten_system_parameters(::NullParameters)
    _flatten_system_parameters(::Type{T}, μ::NamedTuple)
    _flatten_system_parameters(::Type{T}, ::NullParameters)

Return the tuple `(v, layout)` of the flat `Vector` `v` of the system parameters `μ` and the
`ParameterLayout` that puts such a vector back into the shape of `μ`. It is the `flatten` of
`NeuralNetworkParameters` of `μ`, and of the empty `NamedTuple()` for `NullParameters()`.

With `T`, the element type of `v` is `T`; without it, it is the one element type of the leaves of
`μ`, and `Union{}` for `NullParameters()` and `(;)`. `NullParameters()` flattens to an empty
vector. The layout is a value and not a closure, so a layer can store it in a field.

The function throws no error of its own; the errors of `flatten` propagate. It throws an
`ArgumentError` for a leaf of `μ` that `flatten` cannot copy, such as a `String`, and, without `T`,
for leaves of two element types. With `T`, it throws an `InexactError` for a value that `T` cannot
hold, such as `0.5` for `T = Int`.

`HamiltonianNeuralNetworks` calls this function by its qualified name, so it is public.
"""
_flatten_system_parameters(μ::NamedTuple) = flatten(μ)
_flatten_system_parameters(::NullParameters) = flatten(NamedTuple())
_flatten_system_parameters(::Type{T}, μ::NamedTuple) where {T} = flatten(T, μ)
_flatten_system_parameters(::Type{T}, ::NullParameters) where {T} = flatten(T, NamedTuple())

"""
    concatenate_array_with_parameters(x::AbstractArray, ::NullParameters)
    concatenate_array_with_parameters(x::AbstractVector, μ::NamedTuple)
    concatenate_array_with_parameters(x::AbstractMatrix, μ::NamedTuple)
    concatenate_array_with_parameters(x::AbstractArray{<:Any, 3}, μ::NamedTuple)
    concatenate_array_with_parameters(x::AbstractMatrix, μs::AbstractVector)

Return the input of a parametric layer: the state `x` with the flattened system parameters below
it, in the element type of `x`.

- With `NullParameters()`, return `x` itself.
- With one `μ`, flatten it with `_flatten_system_parameters(eltype(x), μ)`, and put it below a
  vector, below every column of a matrix, and below every column of every slice of a 3-tensor. The
  result has as many more rows than `x` as the flat `μ` has entries.
- With a vector `μs`, put `μs[i]` below column `i` of the matrix `x`, and return the columns side
  by side.

Throws a `DimensionMismatch` when the number of columns of `x` is not `length(μs)`, an
`ArgumentError` for an empty `μs` with a matrix of no columns, and an `InexactError` when a
parameter has no value in the element type of `x`, such as `0.5` below an integer state. An error
of `_flatten_system_parameters` propagates. An array of more than three dimensions with a
`NamedTuple`, and a vector with a vector of parameters, have no method.

`HamiltonianNeuralNetworks` calls this function by its qualified name, so it is public.
"""
concatenate_array_with_parameters(x::AbstractArray, ::NullParameters) = x

function concatenate_array_with_parameters(x::AbstractVector, μ::NamedTuple)
    vcat(x, first(_flatten_system_parameters(eltype(x), μ)))
end

function concatenate_array_with_parameters(x::AbstractMatrix, μ::NamedTuple)
    flat = first(_flatten_system_parameters(eltype(x), μ))
    vcat(x, repeat(flat, 1, size(x, 2)))
end

function concatenate_array_with_parameters(x::AbstractArray{<:Any, 3}, μ::NamedTuple)
    flat = first(_flatten_system_parameters(eltype(x), μ))
    cat(x, repeat(reshape(flat, :, 1, 1), 1, size(x, 2), size(x, 3)); dims = 1)
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
`ParametricDataLoader` draws them. A batch holds at least one sample. The output has the
shape of the input.

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
        x::Union{
            AbstractVector, NamedTuple{(:q, :p), <:Tuple{AbstractVector, AbstractVector}}},
        μ::OptionalParameters, ps::NetworkParameters)
    model((x, μ), ps)
end

# A batch maps over the columns of its `d × k` reshape, not over indexed samples: under Zygote, the
# adjoint of each `getindex` allocates a cotangent of the whole batch, so indexed samples cost O(k²).
function apply_parametric(model::Chain, x::AbstractArray{<:Any, 3}, μs::AbstractVector,
        ps::NetworkParameters)
    check_parametric_batch(size(x), μs)
    outputs = map_samples((xᵢ, μ, ps) -> apply_parametric(model, xᵢ, μ, ps), ps,
        sample_columns(x), μs)
    reshape(reduce(hcat, outputs), :, 1, length(μs))
end

# The output of a sample is stacked, `q` above `p`, so that its cotangent is an array, which the rule
# of `map_samples` hands to Zygote as it is.
function apply_parametric(model::Chain, x::NamedTupleOfArrays{<:Any, (:q, :p)},
        μs::AbstractVector, ps::NetworkParameters)
    check_parametric_batch(size(x.q), μs)
    outputs = map_samples(ps, sample_columns(x.q), sample_columns(x.p), μs) do q, p, μ, ps
        output = apply_parametric(model, (q = q, p = p), μ, ps)
        vcat(output.q, output.p)
    end
    stacked = reduce(hcat, outputs)
    d = size(x.q, 1)
    (q = reshape(stacked[1:d, :], :, 1, length(μs)),
        p = reshape(stacked[(d + 1):end, :], :, 1, length(μs)))
end

# The samples of a `d × 1 × k` batch, as the `k` columns of its `d × k` reshape.
sample_columns(x::AbstractArray{<:Any, 3}) = eachcol(reshape(x, size(x, 1), size(x, 3)))

# `f(sample..., ps)` for each sample, where sample `i` is element `i` of each of `samples`.
map_samples(f, ps, samples...) = map((sample...) -> f(sample..., ps), samples...)

# Zygote's adjoint of `map` adds the gradients of the closure it maps, which holds `ps`, from `nothing`,
# so the gradient of `ps` infers as `Union{Nothing, NetworkParameters}`. This rule pulls each sample
# back on its own and adds the gradients of `ps`, as `NetworkParameters`, from the first sample on; a
# batch holds at least one sample. `f` holds the `Chain`, whose trainable parameters are in `ps`, so
# `f` has no gradient.
function ChainRulesCore.rrule(::typeof(map_samples), f, ps::NetworkParameters, samples...)
    outputs_and_pullbacks = map((sample...) -> Zygote.pullback(f, sample..., ps), samples...)
    function map_samples_pullback(Δoutputs)
        Δs = map(((_, pullback), Δ) -> pullback(Δ), outputs_and_pullbacks, unthunk(Δoutputs))
        Δps = sum(Δ -> map_cotangent(storage_gradient, ps, last(Δ)), Δs)
        Δsamples = ntuple(i -> map(Δ -> Δ[i], Δs), length(samples))
        (NoTangent(), NoTangent(), Δps, Δsamples...)
    end
    map(first, outputs_and_pullbacks), map_samples_pullback
end

# A batch holds one time step per sample and one `μ` per sample.
function check_parametric_batch(input_size::NTuple{3, Int}, μs::AbstractVector)
    input_size[2] == 1 && input_size[3] == length(μs) || throw(DimensionMismatch(
        "a batch for `apply_parametric` has one time step and one system parameter per " *
        "sample; the input has $(input_size[2]) time steps and $(input_size[3]) samples, " *
        "and there are $(length(μs)) system parameters"))
    nothing
end
