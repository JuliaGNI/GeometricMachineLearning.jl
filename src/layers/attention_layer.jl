"""
Volume-preserving attention (single heaad attention)

Drawbacks: 
- only works on CPU for now (there is an issue with Enzyme atm)
- the super-fast activation is only implemented in 4d and 6d
"""
struct VolumePreservingAttention{M, N} <: AbstractExplicitLayer{M, N}

    function VolumePreservingAttention(dim::Int)
        new{dim, dim}()
    end
end

function orthonormal_activation_cayley(d::VolumePreservingAttention{4, 4}, A::AbstractArray{T, 3}) where T 
    A_ut = upper_triangular_asymmetrize(A)
    tensor_cayley4(A_ut)
end

function orthonormal_activation_cayley(d::VolumePreservingAttention{6, 6}, A::AbstractArray{T, 3}) where T 
    A_ut = upper_triangular_asymmetrize(A)
    tensor_cayley6(A_ut)
end

function orthonormal_activation_cayley(A::AbstractMatrix{T}) where T 
    reshape(orthonormal_activation_cayley(reshape(A, size(A)..., 1)), size(A)...)
end

function parameterlength(d::VolumePreservingAttention{M, M}) where M
    M * (M-1) ÷ 2
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::VolumePreservingAttention{M, M}; rng::AbstractRNG=Random.default_rng(), initializer!::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    V = KernelAbstractions.allocate(backend, T, parameterlength(d))
    initializer!(rng, V)
    (A = SkewSymMatrix(V, M), )
end

function (d::VolumePreservingAttention{4, 4})(x::AbstractArray{T, 3}, ps::NamedTuple) where {T}
    dim, input_length = size(x)
    @assert dim == 4

    tensor_cayley4(tensor_transpose_tensor_mul(x, mat_tensor_mul(ps.A, x)))
end

function (d::VolumePreservingAttention{6, 6})(x::AbstractArray{T, 3}, ps::NamedTuple) where {T}
    dim, input_length = size(x)
    @assert dim == 6

    tensor_cayley6(tensor_transpose_tensor_mul(x, mat_tensor_mul(ps.A, x)))
end