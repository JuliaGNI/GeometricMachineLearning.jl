# Each of the four layers that take a `Bool` keyword encodes it as a `Bool` type parameter, the same
# kind of parameter `ResNetLayer`, `Dense` and `MultiHeadAttention` use: `skew_sym` in
# `VolumePreservingAttention`, `use_bias` in the feedforward layers, `symmetric` in
# `SymplecticAttention`, and the two init keywords in `SymplecticAutoencoder`. The keyword's value is
# the parameter.
#
# The layers are pinned by their type parameters and by the chains they build, and the keywords are
# `Bool`: a non-`Bool` value is rejected with a `TypeError`.

using GeometricMachineLearning
using GeometricMachineLearning: VolumePreservingFeedForwardLayer
using AbstractNeuralNetworks: GlorotUniform, ZeroInitializer
using Test
import Random

@testset "VolumePreservingAttention encodes skew_sym as a Bool" begin
    @test Tuple(typeof(VolumePreservingAttention(4, 3; skew_sym = true)).parameters) ==
          (4, 4, true, 3)
    @test Tuple(typeof(VolumePreservingAttention(4, 3; skew_sym = false)).parameters) ==
          (4, 4, false, 3)
    @test Tuple(typeof(VolumePreservingAttention(4)).parameters) == (4, 4, false, 0)

    # The parameter is a `Bool` and not a symbol, and the two values are the two weightings.
    @test typeof(VolumePreservingAttention(4, 3; skew_sym = true)).parameters[3] isa Bool
    @test parameterlength(VolumePreservingAttention(4, 3; skew_sym = true)) == 6
    @test parameterlength(VolumePreservingAttention(4, 3; skew_sym = false)) == 16

    # The keyword is a `Bool`, so a number is rejected rather than turned into a parameter.
    @test_throws TypeError VolumePreservingAttention(4, 3; skew_sym = 1)
end

@testset "the feedforward layers are one struct with a triangle parameter" begin
    # One struct, and the two exported names are aliases of it, so `VolumePreservingLowerLayer`
    # names the type in a signature or a `@docs` block.
    @test !isabstracttype(VolumePreservingFeedForwardLayer)
    @test fieldnames(VolumePreservingFeedForwardLayer) == (:activation,)
    @test VolumePreservingLowerLayer{4, 4, false, typeof(identity)} ===
          VolumePreservingFeedForwardLayer{4, 4, false, typeof(identity), :lower}
    @test VolumePreservingUpperLayer{4, 4, true, typeof(identity)} ===
          VolumePreservingFeedForwardLayer{4, 4, true, typeof(identity), :upper}

    lower = VolumePreservingLowerLayer(4, identity; use_bias = false)
    upper = VolumePreservingUpperLayer(4, identity; use_bias = true)
    @test lower isa VolumePreservingFeedForwardLayer{4, 4, false, typeof(identity), :lower}
    @test upper isa VolumePreservingFeedForwardLayer{4, 4, true, typeof(identity), :upper}
    @test lower isa VolumePreservingLowerLayer
    @test upper isa VolumePreservingUpperLayer
    @test lower.activation === identity
    @test typeof(lower).parameters[3] isa Bool

    # The bias keyword is a `Bool` here too.
    @test_throws TypeError VolumePreservingLowerLayer(4, identity; use_bias = 1)

    @test parameterlength(lower) == 6
    @test parameterlength(upper) == 10

    # The two triangles allocate one storage vector and wrap it: the strict lower triangle of a
    # `4 × 4` matrix has six entries, so that is the length for both, and the leaf differs.
    rng() = Random.MersenneTwister(1)
    for T in (Float32, Float64)
        ps_lower = initialparameters(rng(), GlorotUniform(), lower, CPU(), T)
        ps_upper = initialparameters(rng(), GlorotUniform(), upper, CPU(), T)
        @test keys(ps_lower) == (:weight,)
        @test keys(ps_upper) == (:weight, :bias)
        @test length(ps_lower.weight.S) == 6
        @test length(ps_upper.weight.S) == 6
        @test eltype(ps_lower.weight.S) === T
        @test eltype(ps_upper.weight.S) === T
        @test ps_lower.weight isa StrictlyLowerTriangular
        @test ps_upper.weight isa StrictlyUpperTriangular
        @test ps_upper.bias isa Vector{T}

        ps_no_bias = initialparameters(rng(), GlorotUniform(),
            VolumePreservingLowerLayer(4; use_bias = false), CPU(), T)
        @test keys(ps_no_bias) == (:weight,)

        # One name for the bias initialiser: `init_bias`, as in `ResNetLayer` and `sympnets.jl`.
        # The keyword is what a caller reaches, so the test reaches it.
        @test all(iszero, ps_upper.bias)
        drawn = initialparameters(rng(), GlorotUniform(), upper, CPU(), T;
            init_bias = GlorotUniform())
        @test !all(iszero, drawn.bias)
        @test drawn.bias isa Vector{T}
        @test drawn.weight.S == ps_upper.weight.S
    end
    @test_throws MethodError initialparameters(
        rng(), GlorotUniform(), upper, CPU(), Float32; init_bias! = ZeroInitializer())
end

@testset "SymplecticAttention encodes symmetric as a Bool" begin
    for constructor in (SymplecticAttentionQ, SymplecticAttentionP)
        @test typeof(constructor(4)).parameters[4] ===
              GeometricMachineLearning.sa_symmetric_default
        @test typeof(constructor(4; symmetric = true)).parameters[4] === true
        @test typeof(constructor(4; symmetric = false)).parameters[4] === false
        @test_throws TypeError constructor(4; symmetric = 1)
    end
    @test parameterlength(SymplecticAttentionQ(4)) == 3
    @test parameterlength(SymplecticAttentionQ(4; symmetric = false)) == 4
end

@testset "SymplecticAutoencoder encodes both init keywords as Bools" begin
    for encoder_init_q in (true, false), decoder_init_q in (true, false)

        arch = SymplecticAutoencoder(
            6, 4; encoder_init_q, decoder_init_q, n_decoder_blocks = 2)
        @test Tuple(typeof(arch).parameters)[1:2] == (encoder_init_q, decoder_init_q)
    end
    @test_throws TypeError SymplecticAutoencoder(6, 4; encoder_init_q = 1)
    @test_throws TypeError SymplecticAutoencoder(6, 4; decoder_init_q = 1)

    # The four chains, pinned layer by layer: a `:Q` gradient layer, a `:P` one, or the `PSDLayer`
    # that changes the dimension. `encoder_init_q` decides the first gradient layer of each encoder
    # block and `decoder_init_q` the first of each decoder block. One block per dimension step, so
    # with two blocks the chain is one encoder block followed by one decoder block.
    half(l) = l isa PSDLayer ? :psd :
              (l isa GradientLayerQ ? :q : (l isa GradientLayerP ? :p : :other))
    chain_halfs(arch) = Tuple(half(l) for l in Chain(arch).layers)

    kwargs = (n_encoder_layers = 2, n_encoder_blocks = 2, n_decoder_layers = 2,
        n_decoder_blocks = 2)

    @test chain_halfs(SymplecticAutoencoder(
        6, 4; encoder_init_q = true, decoder_init_q = true,
        kwargs...)) == (:q, :p, :psd, :q, :p, :psd)
    @test chain_halfs(SymplecticAutoencoder(
        6, 4; encoder_init_q = true, decoder_init_q = false,
        kwargs...)) == (:q, :p, :psd, :p, :q, :psd)
    @test chain_halfs(SymplecticAutoencoder(
        6, 4; encoder_init_q = false, decoder_init_q = true,
        kwargs...)) == (:p, :q, :psd, :q, :p, :psd)
    @test chain_halfs(SymplecticAutoencoder(
        6, 4; encoder_init_q = false, decoder_init_q = false,
        kwargs...)) == (:p, :q, :psd, :p, :q, :psd)
end

@testset "one encoding for the same idea across the layers" begin
    encodings = (typeof(VolumePreservingAttention(4, 3; skew_sym = true)).parameters[3],
        typeof(VolumePreservingLowerLayer(4; use_bias = true)).parameters[3],
        typeof(SymplecticAttentionQ(4)).parameters[4],
        typeof(SymplecticAutoencoder(6, 4; n_decoder_blocks = 2)).parameters[1],
        typeof(SymplecticAutoencoder(6, 4; n_decoder_blocks = 2)).parameters[2])
    @test all(each -> each isa Bool, encodings)
end
