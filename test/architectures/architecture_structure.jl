# The architectures keep their keywords: a non-default activation reaches every layer that takes
# it, the activation field has one name across the SympNets, the `DataLoader` constructors of
# `LASympNet`, `GSympNet` and `ClassificationTransformer` give the architecture of the dimensions
# they read, and the chain builders that shared a body are one method.

using GeometricMachineLearning
using GeometricMachineLearning: GradientLayer
using AbstractNeuralNetworks: Dense
using Test
import Random

const GML = GeometricMachineLearning

Random.seed!(1234)

include("../helpers/eltypes.jl")

# The methods of `f` whose definitions are in the file `file` under `src/`.
function methods_in(f, file::AbstractString)
    filter(
        m -> endswith(String(m.file), joinpath("src", file)), collect(methods(f)))
end

@testset "LinearSymplecticTransformer keeps a non-default activation" begin
    arch = LinearSymplecticTransformer(4, 3; activation = sin)
    @test arch.activation === sin
    gradient_layers = filter(l -> l isa GradientLayer, collect(Chain(arch).layers))
    @test length(gradient_layers) == 4
    @test all(l -> l.activation === sin, gradient_layers)
end

@testset "StandardTransformerIntegrator stores and uses upscaling_activation" begin
    @test hasfield(StandardTransformerIntegrator, :upscaling_activation)
    arch = StandardTransformerIntegrator(4; transformer_dim = 6, n_heads = 2,
        upscaling_activation = tanh)
    @test arch.upscaling_activation === tanh
    @test Chain(arch).layers[1].σ === Dense(4, 6, tanh).σ

    default = StandardTransformerIntegrator(4; transformer_dim = 6, n_heads = 2)
    @test Chain(default).layers[1].σ === Dense(4, 6, identity).σ
end

@testset "the SympNets name their activation field activation" begin
    @test GSympNet(4; activation = sin).activation === sin
    @test LASympNet(4; activation = sin).activation === sin
    @test !hasfield(GSympNet, :act)
end

@testset "a DataLoader gives the SympNet of its input dimension ($T)" for T in REAL_ELTYPES
    dl = DataLoader(rand(T, 4, 10); suppress_info = true)

    la = LASympNet(dl; depth = 7, nhidden = 2, activation = sin, init_upper_linear = false,
        init_upper_act = false)
    @test la === LASympNet(4; depth = 7, nhidden = 2, activation = sin,
        init_upper_linear = false, init_upper_act = false)
    @test la.depth == 5

    g = GSympNet(dl; n_layers = 3, activation = sin, init_upper = false)
    @test g === GSympNet(4; n_layers = 3, activation = sin, init_upper = false)
    @test g.upscaling_dimension == 8
end

@testset "a DataLoader gives the ClassificationTransformer of its dimensions ($T)" for T in REAL_ELTYPES
    dl = DataLoader(rand(T, 4, 3, 5), rand(T, 2, 1, 5); suppress_info = true)
    @test ClassificationTransformer(dl) ===
          ClassificationTransformer(4, 2, 7, 16, softmax, true, true)
    @test ClassificationTransformer(
        dl; n_heads = 2, L = 3, activation = tanh, Stiefel = false, add_connection = false) ===
          ClassificationTransformer(4, 2, 2, 3, tanh, false, false)
    # The two flags differ, so a constructor that swaps them gives another architecture.
    @test ClassificationTransformer(dl; Stiefel = true, add_connection = false) ===
          ClassificationTransformer(4, 2, 7, 16, softmax, true, false)
    @test ClassificationTransformer(dl; Stiefel = false, add_connection = true) ===
          ClassificationTransformer(4, 2, 7, 16, softmax, false, true)
end

@testset "one chain builder where the bodies were the same" begin
    # One `Chain` method for `LASympNet`, beside the one for `GSympNet`.
    @test length(methods_in(Chain, joinpath("architectures", "sympnet.jl"))) == 2
    # One builder of the alternating gradient layers for GSympNet, LST and ST.
    @test !isdefined(GML, :_make_block_for_initialization)
    @test length(methods(GML._gradient_layers)) == 1
    # One `encoder` and one `decoder` for every autoencoder that is not a `SymplecticAutoencoder`,
    # beside the methods for an arbitrary network.
    @test length(methods_in(encoder, joinpath("architectures", "autoencoder.jl"))) == 2
    @test length(methods_in(decoder, joinpath("architectures", "autoencoder.jl"))) == 2
end

@testset "_gradient_layers alternates from the side that init_upper names" begin
    layers = GML._gradient_layers(4, 8, sin, 3, true)
    @test layers ==
          (GradientLayerQ(4, 8, sin), GradientLayerP(4, 8, sin), GradientLayerQ(4, 8, sin))
    layers = GML._gradient_layers(4, 8, sin, 2, false)
    @test layers == (GradientLayerP(4, 8, sin), GradientLayerQ(4, 8, sin))
    @test GML._gradient_layers(4, 8, sin, 0, true) == ()
end
