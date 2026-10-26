# A keyword default written as a literal and documented through a constant drifts apart silently:
# the docstring interpolates the constant, the caller gets the literal, and nothing compares the
# two. Only a test that reads the default off a constructed object can catch that, because the
# object is the one place both meet. `SymplecticAttentionQ`/`SymplecticAttentionP` and
# `SymplecticTransformer` had drifted that way; the assertions below are what keeps them together.
#
# The layers carry `symmetric` in a type parameter rather than a field, so the default is read off
# the type; the architecture carries it in a field.

using GeometricMachineLearning
using GeometricMachineLearning: Dense, ForcedSympNet, ForcingLayer, ForcingLayerQ,
                                ForcingLayerP, ForcingLayerQP
using Test

GML = GeometricMachineLearning

@testset "ForcingLayer: the constructed default is the documented default" begin
    for constructor in (ForcingLayer, ForcingLayerQ, ForcingLayerP, ForcingLayerQP)
        l = constructor(4)

        # `width = dim`, and the activation is the one the inner network's first layer carries
        @test l.dim === 4
        @test l.width === 4
        @test l.nhidden === GML.f_nhidden_default
        @test l.model.layers[1].σ === Dense(4, 4, GML.f_activation_default).σ
    end

    # the generic constructor's own two keyword defaults
    @test ForcingLayer(4) isa ForcingLayer{4, 4, :P, false}
end

@testset "ForcedSympNet: the constructed default is the documented default" begin
    arch = ForcedSympNet(4)

    @test arch.upscaling_dimension === 2 * arch.dim
    @test arch.n_layers === GML.g_n_layers_default
    @test arch.n_forcing_layers === GML.fs_n_forcing_layers_default
    @test arch.activation === GML.g_activation_default
    @test arch.init_upper === GML.g_init_upper_default
end

@testset "SymplecticAttention: the constructed default is the documented default" begin
    for constructor in (SymplecticAttentionQ, SymplecticAttentionP)
        l = constructor(4)

        # `symmetric` is the fourth type parameter, `:symmetric` or `:arbitrary`.
        expected = GML.sa_symmetric_default ? :symmetric : :arbitrary
        @test typeof(l).parameters[4] === expected

        @test l.activation === GML.sa_activation_default
    end
end

@testset "SymplecticTransformer: the constructed default is the documented default" begin
    arch = SymplecticTransformer(4)

    @test arch.symmetric === GML.st_symmetric_default
    @test arch.init_upper === GML.st_init_upper_default
    @test arch.n_sympnet === GML.st_n_sympnet_default
    @test arch.L === GML.st_L_default
    @test arch.sympnet_activation === GML.st_sympnet_activation_default
    @test arch.attention_activation === GML.st_attention_activation_default
end

@testset "StandardTransformerIntegrator: the constructed default is the documented default" begin
    arch = StandardTransformerIntegrator(4)

    @test arch.n_blocks === GML.sti_n_blocks_default
    @test arch.L === GML.sti_L_default
    @test arch.upscaling_activation === GML.sti_upscaling_activation_default
    @test arch.resnet_activation === GML.sti_resnet_activation_default
    @test arch.attention_activation === GML.sti_attention_activation_default
    @test arch.add_connection === GML.sti_add_connection_default
end

# The exception to the `<prefix>_<keyword>_default` convention: `width` defaults to the `dim` the
# constructor is given, or to `dl.input_dim` in the `DataLoader` constructor, so no module constant
# can hold it and no literal can drift from the docstring. The assertion is that the constructed
# value is the documented one.
@testset "ParametricResNet: the constructed default is the documented default" begin
    arch = GML.ParametricResNet(4)

    @test arch.width == arch.sys_dim
    @test arch.n_blocks === GML.prn_n_blocks_default
    @test arch.activation === GML.prn_activation_default
    @test arch.parameters isa GML.NullParameters
end
