# A keyword default written as a literal and documented through a constant drifts apart silently:
# the docstring interpolates the constant, the caller gets the literal, and nothing compares the
# two. Only a test that reads the default off a constructed object can catch that, because the
# object is the one place both meet. `SymplecticAttentionQ`/`SymplecticAttentionP` and
# `SymplecticTransformer` had drifted that way; the assertions below are what keeps them together.
#
# The layers carry `symmetric` in a type parameter rather than a field, so the default is read off
# the type; the architecture carries it in a field.

using GeometricMachineLearning
using Test

GML = GeometricMachineLearning

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
    @test arch.upsacling_activation === GML.sti_upscaling_activation_default
    @test arch.resnet_activation === GML.sti_resnet_activation_default
    @test arch.attention_activation === GML.sti_attention_activation_default
    @test arch.add_connection === GML.sti_add_connection_default
end
