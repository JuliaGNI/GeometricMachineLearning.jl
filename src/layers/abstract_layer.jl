
@doc raw"""
A `NeuralNetworkLayer` is a map from $\mathbb{R}^{N} \rightarrow \mathbb{R}^{M}$.
"""
abstract type NeuralNetworkLayer{DT,N,M} end

@inline input_size(::NeuralNetworkLayer{DT,N,M}) where {DT,N,M} = N
@inline output_size(::NeuralNetworkLayer{DT,N,M}) where {DT,N,M} = M

function apply!(::AbstractVector, ::AbstractVector, layer::NeuralNetworkLayer)
    error("apply! not implemented for layer type ", typeof(layer))
end

function parameters(layer::NeuralNetworkLayer)
    error("parameters not implemented for layer type ", typeof(layer))
end

(layer::NeuralNetworkLayer)(output, input) = apply!(output, input, layer)
