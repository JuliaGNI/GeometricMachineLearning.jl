@doc raw"""
    VolumePreservingAttention(dim, seq_length)

Make an instance of `VolumePreservingAttention` for a specific dimension and sequence length.

The sequence length is ``0`` by default. This works for all sequence lengths but doesn't apply the super-fast activation and is therefore discouraged.

# Arguments 

The constructor can be called with an optional keyword argument: 
- `skew_sym::Bool`: specifies if the weight matrix is skew symmetric (`true`) or arbitrary (`false`, the default).

# Functor 
Applying a layer of type `VolumePreservingAttention` does the following: 
- First we perform the operation ``X \mapsto X^T A X =: C``, where ``X\in\mathbb{R}^{N\times\mathtt{seq\_length}}`` is a vector containing time series data and ``A`` is the skew symmetric matrix associated with the layer. 
- In a second step we compute the Cayley transform of ``C``; ``\Lambda = \mathrm{Cayley}(C)``.
- The output of the layer is then ``X\Lambda``.

# Implementation

The super fast activation is only implemented for sequence lengths of 2, 3, 4 and 5. 
Other sequence lengths only work on CPU for now (lu decomposition has to be implemented to work for tensors in parallel).
"""
struct VolumePreservingAttention{M, N, ScalarProductType, SL} <: AbstractExplicitLayer{M, N}

    function VolumePreservingAttention(dim::Int, seq_length::Int = 0; skew_sym::Bool=false)
        if skew_sym
            return new{dim, dim, :skew_sym, seq_length}()
        else
            return new{dim, dim, :arbitrary, seq_length}()
        end
    end
end

function orthonormal_activation_cayley(::VolumePreservingAttention, A::AbstractArray{T, 3}) where T 
    cpu_tensor_cayley(A)
end

function orthonormal_activation_cayley(::VolumePreservingAttention{M, M, ScalarProductType, 2}, A::AbstractArray{T, 3}) where {T, M, ScalarProductType} 
    tensor_cayley2(A)
end

function orthonormal_activation_cayley(::VolumePreservingAttention{M, M, ScalarProductType, 3}, A::AbstractArray{T, 3}) where {T, M, ScalarProductType} 
    tensor_cayley3(A)
end

function orthonormal_activation_cayley(::VolumePreservingAttention{M, M, ScalarProductType, 4}, A::AbstractArray{T, 3}) where {T, M, ScalarProductType} 
    tensor_cayley4(A)
end

function orthonormal_activation_cayley(::VolumePreservingAttention{M, M, ScalarProductType, 5}, A::AbstractArray{T, 3}) where {T, M, ScalarProductType} 
    tensor_cayley5(A)
end

# function orthonormal_activation_cayley(A::AbstractMatrix{T}) where T 
#     reshape(orthonormal_activation_cayley(reshape(A, size(A)..., 1)), size(A)...)
# end

function parameterlength(::VolumePreservingAttention{M, M, :skew_sym}) where {M}
    M * (M-1) ÷ 2
end

function parameterlength(::VolumePreservingAttention{M, M, :arbitrary}) where {M}
    M ^2
end

function initialparameters(d::VolumePreservingAttention{M, M, :skew_sym}, backend::KernelAbstractions.Backend, T::Type; rng::AbstractRNG=Random.default_rng(), initializer!::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    V = KernelAbstractions.allocate(backend, T, parameterlength(d))
    initializer!(rng, V)
    (A = SkewSymMatrix(V, M), )
end

function initialparameters(::VolumePreservingAttention{M, M, :arbitrary}, backend::KernelAbstractions.Backend, T::Type; rng::AbstractRNG=Random.default_rng(), initializer!::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    A = KernelAbstractions.allocate(backend, T, M, M)
    initializer!(rng, A)
    (A = A, )
end

function (d::VolumePreservingAttention{M, M, :skew_sym})(x::AbstractArray{T, 3}, ps::NamedTuple) where {T, M}
    tensor_tensor_mul(  x, 
                        orthonormal_activation_cayley(d, 
                            tensor_transpose_tensor_mul(x, mat_tensor_mul(ps.A, x))
                        )
    )
end

function (d::VolumePreservingAttention{M, M, :arbitrary})(x::AbstractArray{T, 3}, ps::NamedTuple) where {T, M}
    x_interim = tensor_mat_skew_sym_assign(x, ps.A) / T(√M)
    tensor_tensor_mul(  x, 
                        orthonormal_activation_cayley(d, x_interim - tensor_transpose(x_interim))
                        )
end

function (d::VolumePreservingAttention)(x::AbstractMatrix, ps::NamedTuple)
    reshape(d(reshape(x, size(x)..., 1), ps), size(x)...)
end