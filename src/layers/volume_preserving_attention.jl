"""
Volume-preserving attention (single heaad attention)

Drawbacks: 
- only works on CPU for now (there is an issue with Enzyme atm)
- the super-fast activation is only implemented for sequence lengths of 4, 5 and 6. We also dispatch over this. 
"""
struct VolumePreservingAttention{M, N, SL} <: AbstractExplicitLayer{M, N}

    function VolumePreservingAttention(dim::Int, seq_length::Int)
        new{dim, dim, seq_length}()
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

function parameterlength(::VolumePreservingAttention{M, M}) where M
    M * (M-1) ÷ 2
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::VolumePreservingAttention{M, M}; rng::AbstractRNG=Random.default_rng(), initializer!::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    V = KernelAbstractions.allocate(backend, T, parameterlength(d))
    initializer!(rng, V)
    (A = SkewSymMatrix(V, M), )
end

@doc raw"""
Here we fist perform the operation 
```math
X \mapsto X^T A X =: C,
```
where ``X\in'mathbb{R}^{N\times\mathtt{seq\_length}}`` is a vector containing time series data and ``A`` is the skew symmetric matrix associated with the layer. 

In a second step we compute the Cayley transform of ``C``. This is the output. 
"""
function (d::VolumePreservingAttention)(x::AbstractArray{T, 3}, ps::NamedTuple) where {T}
    tensor_tensor_mul(  x, 
                        orthonormal_activation_cayley(d, 
                            tensor_transpose_tensor_mul(x, mat_tensor_mul(ps.A, x))
                        )
    )
end