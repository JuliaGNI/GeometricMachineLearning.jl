"""
    ZygotePullback <: AbstractPullback

The pullback based on the [`Zygote`](https://github.com/FluxML/Zygote.jl) backend.

# Examples

For a network that is trained on inputs only, `back(Δ)` is a `NetworkParameters` with the keys of
the parameters the pullback was built from:
```jldoctest
using GeometricMachineLearning

pb = ZygotePullback(AutoEncoderLoss())
nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))
value, back = pb(nn.params, nn.model, rand(10))
g = back(1)
(g isa NetworkParameters, keys(g))

# output

(true, (:L1, :L2))
```
"""
struct ZygotePullback{NNLT} <: AbstractPullback{NNLT}
    loss::NNLT
end

# One method for a network that is trained on inputs only, one for a network that is trained on
# an `(input, output)` pair, and two for a network that also takes the parameters of the system,
# one `μ` or one per sample: the shapes a `NetworkLoss` takes. A method per shape and not a union
# over them, which says nothing about which one arrived.
function (_pullback::ZygotePullback)(ps::NetworkParameters, model, input_nt::ArrayOrNamedTuple)
    value, pb = Zygote.pullback(ps -> _pullback.loss(model, ps, input_nt), ps)
    value, Δ -> only(pb(Δ))
end

function (_pullback::ZygotePullback)(
        ps::NetworkParameters, model,
        input_nt_output_nt::Tuple{<:ArrayOrNamedTuple, <:ArrayOrNamedTuple})
    value, pb = Zygote.pullback(
        ps -> _pullback.loss(model, ps, input_nt_output_nt...), ps)
    value, Δ -> only(pb(Δ))
end

function (_pullback::ZygotePullback)(
        ps::NetworkParameters, model,
        input_output_μ::Tuple{
            <:ArrayOrNamedTuple, <:ArrayOrNamedTuple, <:OptionalParameters})
    value, pb = Zygote.pullback(ps -> _pullback.loss(model, ps, input_output_μ...), ps)
    value, Δ -> only(pb(Δ))
end

function (_pullback::ZygotePullback)(
        ps::NetworkParameters, model,
        input_output_μs::Tuple{<:ArrayOrNamedTuple, <:ArrayOrNamedTuple, <:AbstractVector})
    value, pb = Zygote.pullback(ps -> _pullback.loss(model, ps, input_output_μs...), ps)
    value, Δ -> only(pb(Δ))
end
