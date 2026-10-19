# The architecture constructors that build gradient layers. One `_gradient_layers` builds the
# alternating `Q`/`P` tuple for `GSympNet`, `LASympNet`, `LinearSymplecticTransformer` and
# `SymplecticTransformer`; `LASympNet` has one `Chain` method; the `DataLoader` constructors
# delegate to the dimension ones; the activation field has one name; and `encoder` and `decoder`
# dispatch on the architecture family rather than repeat themselves for `SymplecticCompression`.

using GeometricMachineLearning
using Test
import Random

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
end

@testset "_gradient_layers is the one builder" begin
    @test GML._gradient_layers(4, 8, tanh, 3, true) ==
          (
        GradientLayerQ(4, 8, tanh), GradientLayerP(4, 8, tanh), GradientLayerQ(4, 8, tanh))
    @test GML._gradient_layers(4, 8, tanh, 3, false) ==
          (
        GradientLayerP(4, 8, tanh), GradientLayerQ(4, 8, tanh), GradientLayerP(4, 8, tanh))
    @test GML._gradient_layers(4, 8, tanh, 0, true) == ()

    # The two copies of `_make_block_for_initialization` are gone with it.
    @test !isdefined(GML, :_make_block_for_initialization)

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

@testset "encoder and decoder dispatch on the family" begin
    # The `Unknown…` architecture is the family's: a symplectic compression gets the symplectic one,
    # any other autoencoder the plain one.
    psd = NeuralNetwork(PSDArch(10, 6))
    @test encoder(psd).architecture isa GML.UnknownSymplecticEncoder
    @test decoder(psd).architecture isa GML.UnknownSymplecticDecoder
    @test encoder(psd).model == Chain(PSDLayer(10, 6))
    @test decoder(psd).model == Chain(PSDLayer(6, 10))

    # A `SymplecticAutoencoder` keeps the more specific pair, and that pair is what `ReducedLoss`
    # reaches.
    sae = NeuralNetwork(SymplecticAutoencoder(10, 6))
    @test encoder(sae).architecture isa GML.NonLinearSymplecticEncoder
    @test decoder(sae).architecture isa GML.NonLinearSymplecticDecoder

    plain = NeuralNetwork(Chain(Dense(4, 2, tanh)))
    @test encoder(plain).architecture isa GML.UnknownEncoder
    @test decoder(plain).architecture isa GML.UnknownDecoder
    @test encoder(plain).model === plain.model
end
