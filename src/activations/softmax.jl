abstract type AbstractSoftmax <: AbstractNeuralNetworks.Activation end

@doc raw"""
    VectorSoftmax <: AbstractSoftmax

Turn an arbitrary vector into a probability vector with:

```math
[\mathrm{softmax}(a)]_i = \frac{e^{a_i}}{\sum_{i'=1}^de^{a_i}}. 
```

This is what is most often understood under the name "softmax". [`MatrixSoftmax`](@ref) is the matrix version.
"""
struct VectorSoftmax <: AbstractSoftmax end

@doc raw"""
    MatrixSoftmax

Like [`VectorSoftmax`](@ref) but for matrices:

```math
[\mathrm{softmax}(A)]_{ij} = \frac{e^{A_{ij}}}{\sum_{i'=1, j'=1}^{d,\bar{d}}e^{A_{ij}}}. 
```
"""
struct MatrixSoftmax <: AbstractSoftmax end

(::VectorSoftmax)(x::AbstractArray) = softmax(x)

function (::MatrixSoftmax)(x::AbstractMatrix)
    expX = exp.(x)
    expX / sum(expX)
end

function (::MatrixSoftmax)(x::AbstractArray{<:Number, 3})
    expX = exp.(x)
    ∑expX = sum(expX, dims = (1, 2))
    expX ./ ∑expX
end