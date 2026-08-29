"""
    ZygotePullback <: AbstractPullback

The pullback based on the [`Zygote`](https://github.com/FluxML/Zygote.jl) backend.

# Examples

For a network that is trained on inputs only:
```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: _processing

loss = AutoEncoderLoss()
_pullback = ZygotePullback(loss)
nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))
input = rand(10)
_pullback(nn.params, nn.model, input)[2](1) |> _processing |> typeof

# output

@NamedTuple{L1::@NamedTuple{W::Matrix{Float64}, b::Vector{Float64}}, L2::@NamedTuple{W::Matrix{Float64}, b::Vector{Float64}}}
```

In this example [`_processing`](@ref) is used to get around some `Zygote` quirks.
"""
struct ZygotePullback{NNLT} <: AbstractPullback{NNLT}
    loss::NNLT
end

(_pullback::ZygotePullback)(ps, model, input_nt::QPTOAT)::Tuple = Zygote.pullback(
    ps -> _pullback.loss(model, ps, input_nt), ps)
(_pullback::ZygotePullback)(ps, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT})::Tuple = Zygote.pullback(
    ps -> _pullback.loss(model, ps, input_nt_output_nt...), ps)

"""
    _get_contents(returned_pullback)

Unwrap the single element `Zygote` may wrap a pullback result in.

Together with [`_get_params`](@ref) this makes up [`_processing`](@ref).
"""
# Both shapes, because a pullback produces both. `NeuralNetworkParameters` rewraps the tangent of a
# `NetworkParameters` into one, while a reverse pass seeded with a bare `NamedTuple` hands one back.
_get_contents(nt::NetworkParameters) = nt
_get_contents(nt::NamedTuple) = nt
_get_contents(nt::Tuple{<:Union{NetworkParameters, NamedTuple}}) = nt[1]
function _get_contents(nt::AbstractVector{<:Union{NetworkParameters, NamedTuple}})
    length(nt) == 1 || throw(ArgumentError(
        "the pullback returned $(length(nt)) parameter sets, expected one."))
    nt[1]
end

"""
    _get_params(returned_pullback)

Get the parameters out of a pullback result, whether they come as a
`NetworkParameters`, wrapped in a `NamedTuple` with a single `params` field, or bare.

Together with [`_get_contents`](@ref) this makes up [`_processing`](@ref).
"""
_get_params(nt::NamedTuple) = nt
_get_params(ps::NetworkParameters) = params(ps)
function _get_params(nt::NamedTuple{(:params,), Tuple{AT}}) where {AT}
    @warn "This function was most likely called because @adjoint for `NetworkParameters` hasn't been implemented."
    nt.params
end

"""
    _processing(returned_pullback)

Strip `returned_pullback` from unnecessary `Zygote`-induces garbage.

These two helpers used to be `SymbolicNeuralNetworks._get_params` and
`SymbolicNeuralNetworks._get_contents`; SymbolicNeuralNetworks 0.5 removed them, and they were
never about symbolics in the first place — they clean up what `Zygote` returns.

Also see the docs for [`ZygotePullback`](@ref).
"""
_processing = _get_params ∘ _get_contents
