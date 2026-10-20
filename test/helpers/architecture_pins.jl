# The networks whose outputs `test/architectures/forward_pins.jl` pins, and the three things that
# make a recorded output reproducible: a fixed input, a fixed seed for each network's parameters,
# and a fixed value for each `StiefelManifold` leaf.
#
# A Stiefel leaf is initialised by the QR factorisation of a seeded draw, and two OpenBLAS versions
# give the same draw and different last bits of the factor. Each such leaf is therefore overwritten
# before the pass with a signed selection: column `j` of an `N × n` leaf is `±e_r`, with
# `r = mod(j - 1 + k, N) + 1` for the `k`-th Stiefel leaf of the network. The rows of one leaf are
# distinct, so the leaf is exactly orthonormal, and `k` makes the leaves differ from one another.

using GeometricMachineLearning
using GeometricMachineLearning: params
import Random

# A fixed input with entries in (-1, 1), computed in `Float64` and rounded once to `T`.
function pin_input(::Type{T}, dims::Integer...) where {T}
    T.(reshape([sin(0.7 * i + 0.3) for i in 1:prod(dims)], dims...))
end

foreach_stiefel_leaf(f, W::StiefelManifold) = f(W)
foreach_stiefel_leaf(f, ::AbstractArray) = nothing
function foreach_stiefel_leaf(f, ps::Union{NamedTuple, NetworkParameters})
    for k in keys(ps)
        foreach_stiefel_leaf(f, ps[k])
    end
end

function fix_stiefel_leaves!(ps)
    k = 0
    foreach_stiefel_leaf(ps) do W
        k += 1
        A = W.A
        N, n = size(A)
        fill!(A, zero(eltype(A)))
        for j in 1:n
            A[mod(j - 1 + k, N) + 1, j] = isodd(j + k) ? -one(eltype(A)) : one(eltype(A))
        end
    end
    ps
end

# A network of `T` with the parameters of seed `seed`, its Stiefel leaves fixed.
function pinned_network(model, ::Type{T}, seed::Integer) where {T}
    nn = NeuralNetwork(model, CPU(), T; rng = Random.Xoshiro(seed))
    fix_stiefel_leaves!(params(nn))
    nn
end

# Each case maps a name to a function of the element type that returns the output to pin. The names
# are the keys of the recorded table.
const PIN_CASES = [
    # `GradientLayerQ` and `GradientLayerP`: dense leaves.
    "GSympNet init_upper" =>
        T -> pinned_network(GSympNet(4; n_layers = 3), T, 1)(pin_input(T, 4, 3)),
    "GSympNet init_lower" =>
        T -> pinned_network(
            GSympNet(4; n_layers = 3, init_upper = false), T, 2)(pin_input(T, 4, 3)),
    # `LinearLayer` (a `SymmetricMatrix` leaf), `ActivationLayer` and `BiasLayer`, for every pair of
    # `(init_upper_linear, init_upper_act)`.
    (["LASympNet $(lin) $(act)" =>
          T -> pinned_network(
              LASympNet(
                  4; depth = 3, nhidden = 2, init_upper_linear = lin, init_upper_act = act),
              T, 3)(pin_input(T, 4, 3))
      for lin in (true, false) for act in (true, false)])...,
    # `LinearSymplecticAttention`: a `SymmetricMatrix` leaf.
    "LinearSymplecticTransformer init_upper" =>
        T -> pinned_network(
            LinearSymplecticTransformer(4, 3), T, 4)(pin_input(T, 4, 3)),
    "LinearSymplecticTransformer init_lower" =>
        T -> pinned_network(
            LinearSymplecticTransformer(4, 3; init_upper = false), T, 5)(pin_input(T, 4, 3)),
    # `SymplecticAttention`: a `SymmetricMatrix` leaf and a dense one; with an upscaling, two
    # `PSDLayer`s with Stiefel leaves.
    "SymplecticTransformer symmetric" =>
        T -> pinned_network(
            SymplecticTransformer(4), T, 6)(pin_input(T, 4, 3)),
    "SymplecticTransformer arbitrary" =>
        T -> pinned_network(
            SymplecticTransformer(4; symmetric = false, init_upper = false), T, 7)(pin_input(T, 4, 3)),
    "SymplecticTransformer upscaled" =>
        T -> pinned_network(
            SymplecticTransformer(4; transformer_dim = 6), T, 8)(pin_input(T, 4, 3)),
    # `VolumePreservingLowerLayer` and `VolumePreservingUpperLayer`, with and without a bias:
    # triangular leaves.
    "VolumePreservingFeedForward init_lower" =>
        T -> pinned_network(
            VolumePreservingFeedForward(3, 1, 2), T, 9)(pin_input(T, 3, 3)),
    "VolumePreservingFeedForward init_upper" =>
        T -> pinned_network(
            VolumePreservingFeedForward(3, 1, 2; init_upper = true), T, 10)(pin_input(T, 3, 3)),
    # `VolumePreservingAttention`: a `SkewSymMatrix` leaf and a dense one.
    "VolumePreservingTransformer skew_sym" =>
        T -> pinned_network(
            VolumePreservingTransformer(3, 3; skew_sym = true), T, 11)(pin_input(T, 3, 3)),
    "VolumePreservingTransformer arbitrary" =>
        T -> pinned_network(
            VolumePreservingTransformer(3, 3; skew_sym = false), T, 12)(pin_input(T, 3, 3)),
    # `MultiHeadAttention` with two heads: dense leaves, and Stiefel leaves in the `Transformer`.
    "StandardTransformerIntegrator" =>
        T -> pinned_network(
            StandardTransformerIntegrator(4; n_heads = 2), T, 13)(pin_input(T, 4, 3)),
    "StandardTransformerIntegrator upscaled" =>
        T -> pinned_network(
            StandardTransformerIntegrator(4; transformer_dim = 6, n_heads = 2,
                upscaling_activation = tanh), T, 14)(pin_input(T, 4, 3)),
    "Transformer Stiefel" =>
        T -> pinned_network(
            Transformer(4, 2, 1; Stiefel = true), T, 15)(pin_input(T, 4, 3)),
    # The autoencoders, whole and through the extracted encoder and decoder: `PSDLayer` Stiefel
    # leaves, and the gradient layers of every `(encoder_init_q, decoder_init_q)` pair.
    (["SymplecticAutoencoder $(enc) $(dec) $(part)" =>
          T -> begin
              nn = pinned_network(
                  SymplecticAutoencoder(8, 4; n_encoder_layers = 2,
                      n_decoder_layers = 2, encoder_init_q = enc, decoder_init_q = dec),
                  T,
                  16)
              x = pin_input(T, 8, 2)
              part == "whole" ? nn(x) :
              part == "encoder" ? encoder(nn)(x) :
              decoder(nn)(pin_input(T, 4, 2))
          end
      for enc in (true, false) for dec in (true, false)
      for part in ("whole", "encoder", "decoder")])...,
    (["PSDArch $(part)" =>
          T -> begin
              nn = pinned_network(PSDArch(8, 4), T, 17)
              part == "whole" ? nn(pin_input(T, 8, 2)) :
              part == "encoder" ? encoder(nn)(pin_input(T, 8, 2)) :
              decoder(nn)(pin_input(T, 4, 2))
          end
      for part in ("whole", "encoder", "decoder")])...
]
