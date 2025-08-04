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

function (_pullback::ZygotePullback)(ps, model, input_nt::QPTOAT)::Tuple
    closure = ps -> _pullback.loss(model, ps, input_nt)
    Zygote.pullback(closure, ps)
end
function (_pullback::ZygotePullback)(ps, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT})::Tuple
    closure = ps -> _pullback.loss(model, ps, input_nt_output_nt...)
    Zygote.pullback(closure, ps)
end
function (_pullback::ZygotePullback)(ps, model, input_and_parameters::Tuple{<:QPTOAT, <:QPTOAT, <:NamedTuple})::Tuple
    closure = ps -> _pullback.loss(model, ps, input_and_parameters...)
    Zygote.pullback(closure, ps)
end
function (_pullback::ZygotePullback)(ps, model, input_and_parameters::Tuple{<:QPTOAT, <:QPTOAT, <:AbstractVector})::Tuple
    closure = ps -> _pullback.loss(model, ps, input_and_parameters...)
    Zygote.pullback(closure, ps)
end

"""
    _processing(returned_pullback)

Strip `returned_pullback` from unnecessary `Zygote`-induces garbage.

Also see the docs for [`ZygotePullback`](@ref).
"""
_processing = _get_params∘_get_contents