"""
Classification Layer that takes a matrix as an input and returns a vector that is used for MNIST classification. 
"""
struct Classification{M, N, FT} <: AbstractExplicitLayer{M, N}
    activation::FT
end

function Classification(M::Integer, N::Integer, activation)
    Classification{M, N, typeof(activation)}(activation)
end

function initialparameters(device::KernelAbstractions.Backend, T::Type, ::Classification{M, N}; rng::Random.AbstractRNG=Random.default_rng(), init_weight! = GlorotUniform()) where {M, N}
    weight = KernelAbstractions.allocate(device, T, N, M)
    init_weight!(rng, weight)
    (weight=weight, )
end

function (d::Classification)(output::AbstractArray{T, 3}, ps::NamedTuple) where T 
    d.activation(sum(mat_tensor_mul(ps.weight, output), dims=2)/size(output, 2))
end

function (d::Classification)(output::AbstractArray{T, 2}, ps::NamedTuple) where T 
    d.activation(sum(ps.weight*output, dims=2)/size(output, 2))
end