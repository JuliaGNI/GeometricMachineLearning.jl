@doc raw"""
    VolumePreservingAttention(dim, seq_length)

Make an instance of `VolumePreservingAttention` for a specific dimension and sequence length.

The sequence length is `0` by default, and a layer built with `0` applies to data of any sequence
length: the Cayley transform of the data is one kernel, [`tensor_cayley`](@ref), rather than a
symbolic inverse, and every sequence length runs on the CPU and on a device.

# Arguments 

The constructor can be called with an optional keyword argument: 
- `skew_sym::Bool = false`: specifies if the weight matrix is skew symmetric (`true`) or arbitrary (`false`).

# Functor 
Applying a layer of type `VolumePreservingAttention` does the following: 
- First we perform the operation ``Z \mapsto Z^T A Z =: C``, where ``Z\in\mathbb{R}^{N\times\mathtt{seq\_length}}`` is a vector containing time series data and ``A`` is the skew symmetric matrix associated with the layer (if `skew_sym = true`). 
- In a second step we compute the Cayley transform of ``C``; ``\Lambda = \mathrm{Cayley}(C)``.
- The output of the layer is then ``Z\Lambda``.

# Implementation

The generic activation uses [`tensor_cayley`](@ref), which solves ``(\mathbb{I} + C)\Lambda =
\mathbb{I} - C`` by row-pivoted Gauss–Jordan elimination, one work item per slice. It runs at any
sequence length, on the CPU and on a device, and in the element type of the data.

The layer's `seq_length` chooses the activation: a `seq_length` of 2, 3, 4 or 5 uses the closed form
`tensor_cayley2` to `tensor_cayley5`, whose inverse has been computed symbolically, and every other
`seq_length`, `0` included, uses the kernel, whatever the length of the data the layer is applied to.
"""
struct VolumePreservingAttention{M, N, ScalarProductType, SL} <: AbstractExplicitLayer{M, N}
    function VolumePreservingAttention(dim::Int, seq_length::Int = 0; skew_sym::Bool = false)
        if skew_sym
            return new{dim, dim, :skew_sym, seq_length}()
        else
            return new{dim, dim, :arbitrary, seq_length}()
        end
    end
end

function orthonormal_activation_cayley(::VolumePreservingAttention, A::AbstractArray{
        T, 3}) where {T}
    tensor_cayley(A)
end

function orthonormal_activation_cayley(
        ::VolumePreservingAttention{M, M, ScalarProductType, 2},
        A::AbstractArray{T, 3}) where {T, M, ScalarProductType}
    tensor_cayley2(A)
end

function orthonormal_activation_cayley(
        ::VolumePreservingAttention{M, M, ScalarProductType, 3},
        A::AbstractArray{T, 3}) where {T, M, ScalarProductType}
    tensor_cayley3(A)
end

function orthonormal_activation_cayley(
        ::VolumePreservingAttention{M, M, ScalarProductType, 4},
        A::AbstractArray{T, 3}) where {T, M, ScalarProductType}
    tensor_cayley4(A)
end

function orthonormal_activation_cayley(
        ::VolumePreservingAttention{M, M, ScalarProductType, 5},
        A::AbstractArray{T, 3}) where {T, M, ScalarProductType}
    tensor_cayley5(A)
end

function parameterlength(::VolumePreservingAttention{M, M, :skew_sym}) where {M}
    M * (M-1) ÷ 2
end

function parameterlength(::VolumePreservingAttention{M, M, :arbitrary}) where {M}
    M ^ 2
end

function initialparameters(
        rng::AbstractRNG, initializer!::AbstractNeuralNetworks.Initializer,
        d::VolumePreservingAttention{M, M, :skew_sym},
        backend::KernelAbstractions.Backend, T::Type) where {M}
    V = KernelAbstractions.allocate(backend, T, parameterlength(d))
    initializer!(rng, V)
    (A = SkewSymMatrix(V, M),)
end

function initialparameters(
        rng::AbstractRNG, initializer!::AbstractNeuralNetworks.Initializer,
        ::VolumePreservingAttention{M, M, :arbitrary},
        backend::KernelAbstractions.Backend, T::Type) where {M}
    A = KernelAbstractions.allocate(backend, T, M, M)
    initializer!(rng, A)
    (A = A,)
end

function (d::VolumePreservingAttention{M, M, :skew_sym})(
        x::AbstractArray{
            T, 3}, ps::NamedTuple) where {T, M}
    tensor_tensor_mul(x,
        orthonormal_activation_cayley(d,
            tensor_transpose_tensor_mul(x, _mul(ps.A, x))
        )
    )
end

function (d::VolumePreservingAttention{M, M, :arbitrary})(
        x::AbstractArray{
            T, 3}, ps::NamedTuple) where {T, M}
    tensor_tensor_mul(x, orthonormal_activation_cayley(d, arbitrary_operand(x, ps)))
end

function (d::VolumePreservingAttention)(x::AbstractMatrix, ps::NamedTuple)
    reshape(d(reshape(x, size(x)..., 1), ps), size(x)...)
end

# The operand `C` of the `:arbitrary` path, whose Cayley transform the layer takes: the scalar
# products of the columns of `x` weighted by `A`, divided by `√M`, made skew by one subtraction of a
# permutation of the same array. Both terms are the same array, so `C + Cᵀ` is exactly zero. The
# `:skew_sym` path forms `xᵀAx` instead, which is skew only to rounding.
function arbitrary_operand(x::AbstractArray{T, 3}, ps::NamedTuple) where {T}
    M = size(x, 1)
    x_interim = tensor_mat_skew_sym_assign(x, ps.A) / T(√M)
    x_interim - permutedims(x_interim, (2, 1, 3))
end
