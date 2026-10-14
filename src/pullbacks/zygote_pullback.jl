"""
    ZygotePullback <: AbstractPullback

The pullback based on the [`Zygote`](https://github.com/FluxML/Zygote.jl) backend.

# Examples

For a network that is trained on inputs only:
```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: AutoEncoderLoss, ZygotePullback

loss = AutoEncoderLoss()
_pullback = ZygotePullback(loss)
nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))
input = rand(10)
value, back = _pullback(nn.params, nn.model, input)
back(1) isa NetworkParameters

# output

true
```

The keys of the returned parameter set are those of the parameters the pullback was built from:
```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: AutoEncoderLoss, ZygotePullback

loss = AutoEncoderLoss()
_pullback = ZygotePullback(loss)
nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))
input = rand(10)
value, back = _pullback(nn.params, nn.model, input)
keys(back(1)) == (:L1, :L2)

# output

true
```
"""
struct ZygotePullback{NNLT} <: AbstractPullback{NNLT}
    loss::NNLT
end

# One method for a network that is trained on inputs only, and one for a network that is trained on
# an `(input, output)` pair: the two shapes a `NetworkLoss` takes. A method per shape and not a
# union over them, which says nothing about which one arrived.
function (_pullback::ZygotePullback)(ps::NetworkParameters, model, input_nt::QPTOAT)
    value, pb = Zygote.pullback(ps -> _pullback.loss(model, ps, input_nt), ps)
    value, Δ -> only(pb(Δ))
end

function (_pullback::ZygotePullback)(
        ps::NetworkParameters, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT})
    value, pb = Zygote.pullback(
        ps -> _pullback.loss(model, ps, input_nt_output_nt...), ps)
    value, Δ -> only(pb(Δ))
end
