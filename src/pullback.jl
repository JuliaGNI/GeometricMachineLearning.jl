"""
    AbstractPullback{NNLT<:NetworkLoss}

`AbstractPullback` is an `abstract type` that encompasses all ways of performing differentiation (especially computing the gradient with respect to neural network parameters) in `GeometricMachineLearning`.

If a user wants to implement a custom `Pullback` the following two functions have to be extended:
```julia
(_pullback::AbstractPullback)(ps, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT})
(_pullback::AbstractPullback)(ps, model, input_nt::QPT)
```
based on the `loss::NetworkLoss` that's stored in `_pullback`. Also see [`ZygotePullback`](@ref).
"""
abstract type AbstractPullback{NNLT<:NetworkLoss} end

(_pullback::AbstractPullback)(ps, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT}) = error("Pullback not implemented for input-output pair!")
(_pullback::AbstractPullback)(ps, model, input_nt::QPT) = error("Pullback not implemented for single input!")

"""
    ZygotePullback <: AbstractPullback

The pullback based on the [`Zygote`](https://github.com/FluxML/Zygote.jl) backend.
"""
struct ZygotePullback{NNLT} <: AbstractPullback{NNLT}
    loss::NNLT
end

(_pullback::ZygotePullback)(ps, model, input_nt::QPTOAT) = Zygote.pullback(ps -> _pullback.loss(model, ps, input_nt), ps)
(_pullback::ZygotePullback)(ps, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT}) = Zygote.pullback(ps -> _pullback.loss(model, ps, input_nt_output_nt...), ps)