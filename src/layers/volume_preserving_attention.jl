@doc raw"""
# Volume-preserving attention (single head attention)

Drawbacks: 
- only works on CPU for now (there is an issue with Enzyme atm)
- the super-fast activation is only implemented for sequence lengths of 4, 5 and 6. We also dispatch over this. 

## Constructor 

The constructor is called with: 
- `dim::Int`: The system dimension 
- `seq_length::Int`: The number of time steps that are considered in the attention mechanism. 
- `skew_sym::Bool` (keyword argument): specifies if we the weight matrix is skew symmetric or arbitrary (default is false).

## Functor 
Applying a layer of type `VolumePreservingAttention` does the following: 
- First we perform the operation ``X \mapsto X^T A X =: C``, where ``X\in'mathbb{R}^{N\times\mathtt{seq\_length}}`` is a vector containing time series data and ``A`` is the skew symmetric matrix associated with the layer. 
- In a second step we compute the Cayley transform of ``C``; ``\Lambda = \mathrm{Cayley}(C)``.
- The output of the layer is then ``X\Lambda``.
"""
struct VolumePreservingAttention{M, N, SL, ScalarProductType} <: AbstractExplicitLayer{M, N}

    function VolumePreservingAttention(dim::Int, seq_length::Int; skew_sym::Bool=false)
        if skew_sym
            return new{dim, dim, seq_length, :skew_sym}()
        else
            return new{dim, dim, seq_length, :arbitrary}()
        end
    end
end

function orthonormal_activation_cayley(::VolumePreservingAttention{M, M, 4}, A::AbstractArray{T, 3}) where {T, M} 
    tensor_cayley4(A)
end

function orthonormal_activation_cayley(::VolumePreservingAttention{M, M, 5}, A::AbstractArray{T, 3}) where {T, M} 
    tensor_cayley5(A)
end

function orthonormal_activation_cayley(::VolumePreservingAttention{M, M, 6}, A::AbstractArray{T, 3}) where {T, M} 
    tensor_cayley6(A)
end

function orthonormal_activation_cayley(A::AbstractMatrix{T}) where T 
    reshape(orthonormal_activation_cayley(reshape(A, size(A)..., 1)), size(A)...)
end

function parameterlength(::VolumePreservingAttention{M, M, SL, :skew_sym}) where {M, SL}
    M * (M-1) ÷ 2
end

function parameterlength(::VolumePreservingAttention{M, M, SL, :arbitrary}) where {M, SL}
    M ^2
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::VolumePreservingAttention{M, M, SL, :skew_sym}; rng::AbstractRNG=Random.default_rng(), initializer!::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M, SL}
    V = KernelAbstractions.allocate(backend, T, parameterlength(d))
    initializer!(rng, V)
    (A = SkewSymMatrix(V, M), )
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, ::VolumePreservingAttention{M, M, SL, :arbitrary}; rng::AbstractRNG=Random.default_rng(), initializer!::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M, SL}
    A = KernelAbstractions.allocate(backend, T, M, M)
    initializer!(rng, A)
    (A = A, )
end

function (d::VolumePreservingAttention{M, M, SL, :skew_sym})(x::AbstractArray{T, 3}, ps::NamedTuple) where {T, M, SL}
    tensor_tensor_mul(  x, 
                        orthonormal_activation_cayley(d, 
                            tensor_transpose_tensor_mul(x, mat_tensor_mul(ps.A, x))
                        )
    )
end

function (d::VolumePreservingAttention{M, M, SL, :arbitrary})(x::AbstractArray{T, 3}, ps::NamedTuple) where {T, M, SL}
    x_interm = tensor_mat_skew_sym_assign(x, ps.A) / √M
    tensor_tensor_mul(  x, 
                        orthonormal_activation_cayley(d, 
                            x_interm - tensor_transpose(x_interm) )
                        )
end