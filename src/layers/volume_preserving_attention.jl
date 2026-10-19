@doc raw"""
    VolumePreservingAttention(dim, seq_length)

Make an instance of `VolumePreservingAttention` for a specific dimension and sequence length.

The sequence length is `0` by default, and a layer built with `0` applies to data of any sequence
length.

# Arguments 

The constructor can be called with an optional keyword argument: 
- `skew_sym::Bool = false`: specifies if the weight matrix is skew symmetric (`true`) or arbitrary (`false`).

# Functor 
Applying a layer of type `VolumePreservingAttention` does the following: 
- First we perform the operation ``Z \mapsto Z^T A Z =: C``, where ``Z\in\mathbb{R}^{N\times\mathtt{seq\_length}}`` is a vector containing time series data and ``A`` is the skew symmetric matrix associated with the layer (if `skew_sym = true`). 
- In a second step we compute the Cayley transform of ``C``; ``\Lambda = \mathrm{Cayley}(C)``.
- The output of the layer is then ``Z\Lambda``.

# Implementation

The Cayley transform is computed by [`tensor_cayley`](@ref), which solves ``(\mathbb{I} + C)\Lambda =
\mathbb{I} - C`` by row-pivoted Gauss–Jordan elimination at any sequence length, on the CPU and on a
device, and in the element type of the data.
"""
struct VolumePreservingAttention{M, N, SkewSym, SL} <: AbstractExplicitLayer{M, N}
    function VolumePreservingAttention(dim::Int, seq_length::Int = 0; skew_sym::Bool = false)
        new{dim, dim, skew_sym, seq_length}()
    end
end

function orthonormal_activation_cayley(::VolumePreservingAttention, A::AbstractArray{
        T, 3}) where {T}
    tensor_cayley(A)
end

function parameterlength(::VolumePreservingAttention{M, M, true}) where {M}
    M * (M-1) ÷ 2
end

function parameterlength(::VolumePreservingAttention{M, M, false}) where {M}
    M ^ 2
end

function initialparameters(
        rng::AbstractRNG, initializer!::AbstractNeuralNetworks.Initializer,
        d::VolumePreservingAttention{M, M, true},
        backend::KernelAbstractions.Backend, ::Type{T}) where {M, T}
    V = KernelAbstractions.allocate(backend, T, parameterlength(d))
    initializer!(rng, V)
    (A = SkewSymMatrix(V, M),)
end

function initialparameters(
        rng::AbstractRNG, initializer!::AbstractNeuralNetworks.Initializer,
        ::VolumePreservingAttention{M, M, false},
        backend::KernelAbstractions.Backend, ::Type{T}) where {M, T}
    A = KernelAbstractions.allocate(backend, T, M, M)
    initializer!(rng, A)
    (A = A,)
end

function (d::VolumePreservingAttention{M, M, true})(
        x::AbstractArray{
            T, 3}, ps::NamedTuple) where {T, M}
    tensor_tensor_mul(x,
        orthonormal_activation_cayley(d,
            tensor_transpose_tensor_mul(x, _mul(ps.A, x))
        )
    )
end

function (d::VolumePreservingAttention{M, M, false})(
        x::AbstractArray{
            T, 3}, ps::NamedTuple) where {T, M}
    tensor_tensor_mul(x, orthonormal_activation_cayley(d, arbitrary_operand(x, ps)))
end

function (d::VolumePreservingAttention)(x::AbstractMatrix, ps::NamedTuple)
    reshape(d(reshape(x, size(x)..., 1), ps), size(x)...)
end

# The operand `C` of the `skew_sym = false` path, whose Cayley transform the layer takes: the scalar
# products of the columns of `x` weighted by `A`, divided by `√M`, made skew by one subtraction of a
# permutation of the same array. Both terms are the same array, so `C + Cᵀ` is exactly zero. The
# `skew_sym = true` path forms `xᵀAx` instead, which is skew only to rounding.
function arbitrary_operand(x::AbstractArray{T, 3}, ps::NamedTuple) where {T}
    M = size(x, 1)
    x_interim = tensor_mat_skew_sym_assign(x, ps.A) / T(√M)
    x_interim - permutedims(x_interim, (2, 1, 3))
end
