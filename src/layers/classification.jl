@doc raw"""
    ClassificationLayer(input_dim, output_dim, activation)

Make an instance of `ClassificationLayer`.

`ClassificationLayer` takes a matrix as an input and returns a vector that is used for MNIST classification. 

# Arguments

`ClassificationLayer` has the following optional keyword argument: 
- `average:Bool=false`.

If this keyword argument is set to `true`, then the output is computed as 
```math
    \frac{1}{N}\sum_{i=1}^N[input]_{\bullet{}i}.
``` 

If set to `false` (the default) it picks the last column of the input. 
"""
struct ClassificationLayer{M, N, average, FT} <: AbstractExplicitLayer{M, N}
    activation::FT
end

function ClassificationLayer(input_dim::Integer, output_dim::Integer, activation; average::Bool=false)
    ClassificationLayer{input_dim, output_dim, average, typeof(activation)}(activation)
end

function initialparameters(::ClassificationLayer{M, N}, device::KernelAbstractions.Backend, T::Type; rng::Random.AbstractRNG=Random.default_rng(), init_weight! = GlorotUniform()) where {M, N}
    weight = KernelAbstractions.allocate(device, T, N, M)
    init_weight!(rng, weight)
    (weight=weight, )
end

function (d::ClassificationLayer{M, N, true})(output::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T} 
    d.activation(sum(mat_tensor_mul(ps.weight, output), dims=2)/size(output, 2))
end

function (d::ClassificationLayer{M, N, true})(output::AbstractArray{T, 2}, ps::NamedTuple) where {M, N, T} 
    d.activation(sum(ps.weight*output, dims=2)/size(output, 2))
end

function (d::ClassificationLayer{M, N, false})(output::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T} 
    d.activation(assign_output_estimate(mat_tensor_mul(ps.weight, output), 1))
end

function (d::ClassificationLayer{M, N, false})(output::AbstractArray{T, 2}, ps::NamedTuple) where {M, N, T} 
    d.activation(ps.weight*output[:,end:end])
end