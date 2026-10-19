# The architecture constructors that build gradient layers. One `_gradient_layers` builds the
# alternating `Q`/`P` tuple for `GSympNet`, `LinearSymplecticTransformer` and
# `SymplecticTransformer`; `LASympNet` builds its chain of linear, bias and activation layers in its
# one `Chain` method; the `DataLoader` constructors delegate to the dimension ones; the activation
# field has one name; and `encoder` and `decoder` dispatch on the architecture family rather than
# repeat themselves for `SymplecticCompression`.

using GeometricMachineLearning
using Test
import Random

include("../helpers/eltypes.jl")

GML = GeometricMachineLearning

Random.seed!(1234)

# The layers a `Chain` is expected to hold, from the letters of its layer sequence: `q`/`p` a linear
# layer acting on that half, `aq`/`ap` an activation layer, `b` the bias layer. The letters are
# written out at each pin; this maps them to the layers.
function layer_of(letter::Symbol, dim::Integer)
    letter === :q && return LinearLayerQ(dim)
    letter === :p && return LinearLayerP(dim)
    letter === :aq && return ActivationLayerQ(dim, tanh)
    letter === :ap && return ActivationLayerP(dim, tanh)
    letter === :b && return GML.BiasLayer(dim)
    error("no layer for $letter")
end

expected_layers(letters, dim) = Tuple(layer_of(letter, dim) for letter in letters)

@testset "LASympNet's four chains are one method" begin
    # `depth = 2` and `nhidden = 2`: each hidden block is two linear layers, the bias and the two
    # activation layers; the chain then closes with two linear layers and a bias.
    blocks(first_linear, first_activation) = (
        first_linear, first_linear === :q ? :p : :q, :b,
        first_activation, first_activation === :aq ? :ap : :aq)
    pinned(first_linear, first_activation) = (
        repeat(collect(blocks(first_linear,
                first_activation)), 2)..., first_linear,
        first_linear === :q ? :p : :q, :b)

    for init_upper_linear in (true, false), init_upper_act in (true, false)

        arch = LASympNet(4; depth = 2, nhidden = 2, init_upper_linear, init_upper_act)
        letters = pinned(init_upper_linear ? :q : :p, init_upper_act ? :aq : :ap)
        @test Tuple(Chain(arch).layers) == expected_layers(letters, 4)
    end

    # The four chains are distinct from each other, so the pin above is not one chain four times.
    chains = [typeof(Chain(LASympNet(4; depth = 2, nhidden = 2,
                  init_upper_linear, init_upper_act)).layers)
              for init_upper_linear in (true, false) for init_upper_act in (true, false)]
    @test length(unique(chains)) == 4

    # And the one method gives the recorded numbers, bit for bit: the output of `Chain(arch)` on a
    # fixed input with a seeded parameter draw, for all four init combinations and both precisions.
    # A change of the layer order or of a layer's half would keep the type pins above and change
    # these.
    recorded = Dict(
        Float32 => Dict(
            (false, false) => Float32[0.6712964, -7.429632, -1.3637329, -4.1821103],
            (false, true) => Float32[1.645875, -3.3657727, -0.62259567, -1.9627066],
            (true, false) => Float32[-0.24205601, -6.9744425, -2.618807, -9.395262],
            (true, true) => Float32[-0.4790964, -6.646101, -1.8006339, -8.6097145]),
        Float64 => Dict(
            (false, false) => [0.8941395918067143, -0.15713251495323832,
                0.5244622422207745, 0.020362058891310042],
            (false, true) => [5.557287751876066, 1.6364188650827753,
                -0.770298151205102, 2.6477895852312754],
            (true, false) => [-0.6688705250610155, 2.379906628391023,
                5.361852478306595, 3.809708762796686],
            (true, true) => [1.0159831533442056, 2.220683843079769, 4.014010476373879,
                5.307298641957572]))
    for T in REAL_ELTYPES, (init_upper_linear, init_upper_act) in keys(recorded[T])

        Random.seed!(1234)
        nn = NeuralNetwork(
            Chain(LASympNet(4; depth = 2, nhidden = 1,
                init_upper_linear, init_upper_act)), T)
        y = nn(T[0.25, -0.5, 1.0, 0.75])
        @test y == recorded[T][(init_upper_linear, init_upper_act)]
        @test eltype(y) === T
    end
end

@testset "_gradient_layers is the one builder" begin
    @test GML._gradient_layers(4, 8, tanh, 3, true) ==
          (
        GradientLayerQ(4, 8, tanh), GradientLayerP(4, 8, tanh), GradientLayerQ(4, 8, tanh))
    @test GML._gradient_layers(4, 8, tanh, 3, false) ==
          (
        GradientLayerP(4, 8, tanh), GradientLayerQ(4, 8, tanh), GradientLayerP(4, 8, tanh))
    @test GML._gradient_layers(4, 8, tanh, 0, true) == ()

    # `GSympNet`
    @test Chain(GSympNet(4; n_layers = 3, init_upper = true)).layers ==
          GML._gradient_layers(4, 8, tanh, 3, true)
    @test Chain(GSympNet(4; n_layers = 3, init_upper = false, activation = sin)).layers ==
          GML._gradient_layers(4, 8, sin, 3, false)

    # `LinearSymplecticTransformer`
    @test Chain(LinearSymplecticTransformer(4, 3; n_sympnet = 2, L = 1)).layers ==
          (LinearSymplecticAttentionQ(4, 3), GradientLayerQ(4, 8, tanh),
        GradientLayerP(4, 8, tanh), LinearSymplecticAttentionP(4, 3),
        GradientLayerQ(4, 8, tanh), GradientLayerP(4, 8, tanh))

    # `SymplecticTransformer`
    @test Chain(SymplecticTransformer(4; n_sympnet = 2, L = 1)).layers ==
          (SymplecticAttentionQ(4), GradientLayerQ(4, 8, tanh), GradientLayerP(4, 8, tanh),
        SymplecticAttentionP(4), GradientLayerQ(4, 8, tanh), GradientLayerP(4, 8, tanh))
end

@testset "the DataLoader constructors delegate to the dimension ones" begin
    dl = DataLoader(rand(4, 20); suppress_info = true)
    same_fields(a, b) = typeof(a) === typeof(b) &&
                        all(f -> getfield(a, f) == getfield(b, f), fieldnames(typeof(a)))

    @test same_fields(LASympNet(dl), LASympNet(dl.input_dim))
    @test same_fields(LASympNet(dl; depth = 2, init_upper_act = false),
        LASympNet(dl.input_dim; depth = 2, init_upper_act = false))
    @test same_fields(GSympNet(dl), GSympNet(dl.input_dim))
    @test same_fields(GSympNet(dl; n_layers = 3), GSympNet(dl.input_dim; n_layers = 3))
    @test GSympNet(dl).upscaling_dimension == 2 * dl.input_dim
end

@testset "the activation field has one name" begin
    @test hasfield(GSympNet, :activation)
    @test !hasfield(GSympNet, :act)
    @test GSympNet(4).activation === GeometricMachineLearning.g_activation_default
    @test LASympNet(4).activation === GeometricMachineLearning.la_activation_default
end

# A plain autoencoder: an `AutoEncoder` that is not a `SymplecticCompression`. The package exports
# no such architecture, and the extraction contract is stated for the whole `AutoEncoder` family, so
# the test carries the smallest one that the family's own defaults can work with.
struct PlainAE <: GML.AutoEncoder
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
    n_decoder_blocks::Int
end

function GML.encoder_layers_from_iteration(::PlainAE, iterations::AbstractVector{<:Integer})
    Tuple(Dense(iterations[i], iterations[i + 1], tanh) for i in 1:(length(iterations) - 1))
end

function GML.decoder_layers_from_iteration(::PlainAE, iterations::AbstractVector{<:Integer})
    Tuple(Dense(iterations[i], iterations[i + 1], tanh) for i in 1:(length(iterations) - 1))
end

@testset "encoder and decoder dispatch on the family ($T)" for T in REAL_ELTYPES
    # The `Unknown…` architecture is the family's: a symplectic compression gets the symplectic pair,
    # a `SymplecticAutoencoder` its own more specific pair, and any other autoencoder the plain pair.
    psd = NeuralNetwork(PSDArch(10, 6), T)
    @test encoder(psd).architecture isa GML.UnknownSymplecticEncoder
    @test decoder(psd).architecture isa GML.UnknownSymplecticDecoder
    @test encoder(psd).model == Chain(PSDLayer(10, 6))
    @test decoder(psd).model == Chain(PSDLayer(6, 10))

    # A `SymplecticAutoencoder` keeps the more specific pair, and that pair is what `ReducedLoss`
    # reaches.
    sae = NeuralNetwork(SymplecticAutoencoder(10, 6), T)
    @test encoder(sae).architecture isa GML.NonLinearSymplecticEncoder
    @test decoder(sae).architecture isa GML.NonLinearSymplecticDecoder

    plain = NeuralNetwork(PlainAE(4, 2, 2, 2), T)
    @test encoder(plain).architecture isa GML.UnknownEncoder
    @test decoder(plain).architecture isa GML.UnknownDecoder
    @test encoder(plain).model == Chain(Dense(4, 2, tanh))
    @test decoder(plain).model == Chain(Dense(2, 4, tanh))
end

# The recorded outputs of the extracted networks, with `Random.seed!(1234)` drawn before each
# construction. A family whose extraction read a different layer or parameter set would keep the
# architecture pins above and change these.
#
# A `StiefelManifold` leaf is the CholeskyQR2 of a seeded draw, and the OpenBLAS of Julia 1.12.7 and
# 1.13.2 (0.3.29 and 0.3.30) factorises the same draw to different last bits, so every leaf is
# overwritten with a fixed selection before the pinned pass. The four vectors below are identical on
# the two versions.
const SAE_ENCODER = Dict(
    Float32 => Float32[-0.14297527, -1.047717, -1.1896557, -0.8492112],
    Float64 => [-0.3912090654971454, -1.1381624844184113, -0.9884120694402656,
        -0.6646612400702108])
const SAE_DECODER = Dict(
    Float32 => Float32[-0.18219787, 0.0, -0.10663684, -0.4, 0.0, -0.3],
    Float64 => [-0.25069325449750246, 0.0, -0.17555611823221984, -0.4, 0.0, -0.3])
const PSD_ENCODER = Dict(Float32 => Float32[0.25, 0.5, 0.75, 1.25],
    Float64 => [0.25, 0.5, 0.75, 1.25])
const PSD_DECODER = Dict(Float32 => Float32[0.0, -0.1, -0.2, 0.0, -0.3, -0.4],
    Float64 => [0.0, -0.1, -0.2, 0.0, -0.3, -0.4])

# Column `j` is `±e[mod(j - 1 + which, rows) + 1]`, so the columns are exactly orthonormal for every
# `which`, and the four leaves below get four different selections.
function signed_selection(::Type{T}, rows::Integer, cols::Integer, which::Integer) where {T}
    S = zeros(T, rows, cols)
    for j in 1:cols
        S[mod(j - 1 + which, rows) + 1, j] = isodd(which + j) ? -one(T) : one(T)
    end
    S
end

# One selection per leaf, in parameter order: the SAE's encoder and decoder weight, then the
# `PSDArch`'s.
function overwrite_stiefels!(counter::Base.RefValue{Int}, Y::StiefelManifold{T}) where {T}
    counter[] += 1
    Y.A .= signed_selection(T, size(Y.A, 1), size(Y.A, 2), counter[])
    nothing
end
function overwrite_stiefels!(counter::Base.RefValue{Int}, x::NamedTuple)
    (foreach(y -> overwrite_stiefels!(counter, y), values(x)); nothing)
end
function overwrite_stiefels!(counter::Base.RefValue{Int}, x::GML.NetworkParameters)
    overwrite_stiefels!(counter, GML.params(x))
end
overwrite_stiefels!(::Base.RefValue{Int}, x) = nothing

fixed_leaves!(counter, nn) = (overwrite_stiefels!(counter, GML.params(nn)); nothing)

@testset "the extracted networks give the recorded numbers ($T)" for T in REAL_ELTYPES
    x = T[0.25, -0.5, 1.0, 0.75, -1.25, 0.5]
    reduced = T[0.1, -0.2, 0.3, -0.4]

    # One counter over both networks, so that no two leaves of the test carry the same selection.
    leaves = Ref(0)

    Random.seed!(1234)
    sae = NeuralNetwork(
        SymplecticAutoencoder(6, 4;
            n_encoder_blocks = 2, n_decoder_blocks = 2), T)
    fixed_leaves!(leaves, sae)
    encoded, decoded = encoder(sae)(x), decoder(sae)(reduced)
    @test encoded == SAE_ENCODER[T]
    @test decoded == SAE_DECODER[T]

    Random.seed!(1234)
    psd = NeuralNetwork(PSDArch(6, 4), T)
    fixed_leaves!(leaves, psd)
    @test encoder(psd)(x) == PSD_ENCODER[T]
    @test decoder(psd)(reduced) == PSD_DECODER[T]

    for y in (encoded, decoded, encoder(psd)(x), decoder(psd)(reduced))
        @test y isa AbstractVector{T}
        @test eltype(y) === T
    end
end
