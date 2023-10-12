@doc raw"""
Classification Layer that takes a matrix as an input and returns a vector that is used for MNIST classification. 

It has the following arguments: 
- `M`: input dimension 
- `N`: output dimension 
- `activation`: the activation function 

And the following optional argument: 
- `average`: If this is set to `true`, then the output is computed as $\frac{1}{N}\sum_{i=1}^N[input]_{\bullet{}i}$. If set to `false` (the default) it picks the last column of the input. 
"""
struct Classification{M, N, average, FT} <: AbstractExplicitLayer{M, N}
    activation::FT
end

function Classification(M::Integer, N::Integer, activation; average::Bool=false)
    Classification{M, N, average, typeof(activation)}(activation)
end

function initialparameters(device::KernelAbstractions.Backend, T::Type, ::Classification{M, N}; rng::Random.AbstractRNG=Random.default_rng(), init_weight! = GlorotUniform()) where {M, N}
    weight = KernelAbstractions.allocate(device, T, N, M)
    init_weight!(rng, weight)
    (weight=weight, )
end

function (d::Classification{M, N, true})(output::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T} 
    d.activation(sum(mat_tensor_mul(ps.weight, output), dims=2)/size(output, 2))
end

function (d::Classification{M, N, true})(output::AbstractArray{T, 2}, ps::NamedTuple) where {M, N, T} 
    d.activation(sum(ps.weight*output, dims=2)/size(output, 2))
end

function (d::Classification{M, N, false})(output::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T} 
    d.activation(assign_output_estimate(mat_tensor_mul(ps.weight, output), 1))
end

function (d::Classification{M, N, false})(output::AbstractArray{T, 2}, ps::NamedTuple) where {M, N, T} 
    d.activation(ps.weight*output[:,end:end])
end