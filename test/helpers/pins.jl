# The cases of the pinned forward passes, and the comparison that a pin uses.
#
# A pin is the output of one forward pass of a seeded network on a fixed input, recorded as a
# literal from commit 4c4a8320 with the cases and the seeds below. The literals are in
# `test/helpers/pinned_outputs.jl`.
#
# Two things keep a pin independent of the platform:
# - the input is a closed form, `sin(i + 2j)`, and not a random draw;
# - every `StiefelManifold` leaf is overwritten with a fixed signed selection of columns of the
#   identity before the pass. Its seeded initialisation is orthonormalised through BLAS, whose last
#   bits differ between OpenBLAS versions.
# The comparison is at 16 ULPs of the largest magnitude of the pinned vector. A value that passes
# through GEMM and `tanh` differs by a few ULPs between x86_64 and arm64, and a wrong layer moves it
# by a fraction. The scale is the vector's and not each entry's: an entry that comes out of
# cancellation, such as -0.0686 beside 1.276 in the deep `SymplecticAutoencoder` pass, carries the
# rounding error of the terms it came from, and floating point gives it no accuracy relative to its
# own magnitude.

using GeometricMachineLearning
using GeometricMachineLearning: params, ResNetLayer
using NeuralNetworkParameters: NetworkParameters
import Random

const PIN_ULPS = 16

function pin_input(::Type{T}, dim::Integer, cols::Integer) where {T}
    T[sin(i + 2j)
      for i in 1:dim, j in 1:cols]
end

# The deviation of `a` from the pin `b`, in ULPs of the largest magnitude of `b`: the largest
# `abs(x - y)` over the entries, divided by `eps(maximum(abs, b))`. `Inf` when the sizes differ.
function pin_ulps(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || return Inf
    maximum(abs(x - y) for (x, y) in zip(a, b)) / eps(maximum(abs, b))
end

# A type as its name and its parameters, so that a pin of a type does not depend on the alias that
# `show` picks for it.
type_string(t::DataType) = string(nameof(t), "{", join(repr.(t.parameters), ", "), "}")

# Overwrite `A` with the signed columns `±e_{j + shift}` of the identity, so that `A` has exactly
# orthonormal columns and a different pattern for each `shift`.
function fixed_orthonormal!(A::AbstractMatrix, shift::Integer)
    n, k = size(A)
    @assert k ≤ n
    fill!(A, zero(eltype(A)))
    for j in 1:k
        A[mod1(j + shift, n), j] = iseven(j + shift) ? one(eltype(A)) : -one(eltype(A))
    end
    A
end

function _fix_stiefel!(x::StiefelManifold, count::Base.RefValue{Int})
    fixed_orthonormal!(
        x.A, count[] += 1)
end
function _fix_stiefel!(x::Union{NamedTuple, NetworkParameters}, count::Base.RefValue{Int})
    for k in keys(x)
        _fix_stiefel!(x[k], count)
    end
end
_fix_stiefel!(::Any, ::Base.RefValue{Int}) = nothing

# Overwrite every Stiefel leaf of `ps`, each with its own pattern, in the order of the keys.
fix_stiefel_leaves!(ps) = (_fix_stiefel!(ps, Ref(0)); ps)

# A seeded network of element type `T` for `model`, with its Stiefel leaves fixed.
function pinned_network(model, ::Type{T}, seed::Integer) where {T}
    Random.seed!(seed)
    nn = NeuralNetwork(model, CPU(), T)
    fix_stiefel_leaves!(params(nn))
    nn
end

# The seed of the `i`-th case of each list below.
layer_seed(i::Integer) = 1000 + i
architecture_seed(i::Integer) = 2000 + i
autoencoder_seed(i::Integer) = 3000 + i

# Each case is `(name, model, input dimension, number of columns)`. A layer is wrapped in a `Chain`.
function layer_pin_cases()
    (("GradientLayerQ", Chain(GradientLayerQ(4, 6, tanh)), 4, 3),
        ("GradientLayerP", Chain(GradientLayerP(4, 6, tanh)), 4, 3),
        ("ActivationLayerQ", Chain(ActivationLayerQ(4, tanh)), 4, 3),
        ("ActivationLayerP", Chain(ActivationLayerP(4, tanh)), 4, 3),
        ("LinearLayerQ", Chain(LinearLayerQ(4)), 4, 3),
        ("LinearLayerP", Chain(LinearLayerP(4)), 4, 3),
        ("SymplecticAttentionQ symmetric",
            Chain(SymplecticAttentionQ(4; symmetric = true)), 4, 3),
        ("SymplecticAttentionP symmetric",
            Chain(SymplecticAttentionP(4; symmetric = true)), 4, 3),
        ("SymplecticAttentionQ arbitrary",
            Chain(SymplecticAttentionQ(4; symmetric = false)), 4, 3),
        ("SymplecticAttentionP arbitrary",
            Chain(SymplecticAttentionP(4; symmetric = false)), 4, 3),
        ("LinearSymplecticAttentionQ", Chain(LinearSymplecticAttentionQ(4, 3)), 4, 3),
        ("LinearSymplecticAttentionP", Chain(LinearSymplecticAttentionP(4, 3)), 4, 3),
        ("VolumePreservingLowerLayer bias",
            Chain(VolumePreservingLowerLayer(4, tanh; use_bias = true)), 4, 3),
        ("VolumePreservingLowerLayer no bias",
            Chain(VolumePreservingLowerLayer(4, tanh; use_bias = false)), 4, 3),
        ("VolumePreservingUpperLayer bias",
            Chain(VolumePreservingUpperLayer(4, tanh; use_bias = true)), 4, 3),
        ("VolumePreservingUpperLayer no bias",
            Chain(VolumePreservingUpperLayer(4, tanh; use_bias = false)), 4, 3),
        ("VolumePreservingAttention skew_sym",
            Chain(VolumePreservingAttention(4, 3; skew_sym = true)), 4, 3),
        ("VolumePreservingAttention arbitrary",
            Chain(VolumePreservingAttention(4, 3; skew_sym = false)), 4, 3),
        ("MultiHeadAttention 1 head", Chain(MultiHeadAttention(4, 1)), 4, 3),
        ("MultiHeadAttention 2 heads", Chain(MultiHeadAttention(4, 2)), 4, 3),
        ("MultiHeadAttention 1 head Stiefel",
            Chain(MultiHeadAttention(4, 1; Stiefel = true)), 4, 3),
        ("MultiHeadAttention 2 heads Stiefel",
            Chain(MultiHeadAttention(4, 2; Stiefel = true)), 4, 3),
        ("ResNetLayer bias", Chain(ResNetLayer(4, tanh; use_bias = true)), 4, 3),
        ("ResNetLayer no bias", Chain(ResNetLayer(4, tanh; use_bias = false)), 4, 3),
        ("PSDLayer down", Chain(PSDLayer(4, 2)), 4, 3),
        ("PSDLayer up", Chain(PSDLayer(2, 4)), 2, 3),
        ("PositionalEncoding", Chain(PositionalEncoding(4)), 4, 3))
end

function architecture_pin_cases()
    (("GSympNet upper", GSympNet(4), 4, 3),
        ("GSympNet lower, 3 layers", GSympNet(4; n_layers = 3, init_upper = false), 4, 3),
        ("LinearSymplecticTransformer upper", LinearSymplecticTransformer(4, 3), 4, 3),
        ("LinearSymplecticTransformer lower",
            LinearSymplecticTransformer(4, 3; init_upper = false, n_sympnet = 3), 4, 3),
        ("SymplecticTransformer", SymplecticTransformer(4), 4, 3),
        ("SymplecticTransformer upscaled, arbitrary, lower",
            SymplecticTransformer(4; transformer_dim = 6, symmetric = false, init_upper = false),
            4, 3),
        ("StandardTransformerIntegrator 2 heads",
            StandardTransformerIntegrator(4; n_heads = 2), 4, 3),
        ("StandardTransformerIntegrator upscaled",
            StandardTransformerIntegrator(4; transformer_dim = 6, n_heads = 2), 4, 3),
        ("VolumePreservingFeedForward", VolumePreservingFeedForward(4), 4, 3),
        ("ResNet", ResNet(4, 2, tanh), 4, 3),
        ("VolumePreservingTransformer", VolumePreservingTransformer(4, 3), 4, 3),
        ("SymplecticAutoencoder", SymplecticAutoencoder(8, 4), 8, 3),
        ("PSDArch", PSDArch(8, 4), 8, 3))
end

# Each case is `(name, architecture, full dimension, reduced dimension)`; the pins are the outputs of
# `encoder(nn)` on the full input and of `decoder(nn)` on the reduced input.
function autoencoder_pin_cases()
    (("SymplecticAutoencoder", SymplecticAutoencoder(8, 4), 8, 4),
        ("PSDArch", PSDArch(8, 4), 8, 4))
end
