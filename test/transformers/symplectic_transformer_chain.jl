# `Chain(::SymplecticTransformer)` has one method per upscaling shape, and each reaches a different
# builder. `:NoUpscale` goes to `create_layers_for_transformer_dimension_equal_system_dimension`;
# `:Upscale` goes to the unequal-dimension builder, which wraps that one in a `PSDLayer` on each
# side. The expected counts below are read off those two functions, not off a run.
#
# The equal-dimension builder emits, for each of the `L` blocks: one `SymplecticAttentionQ`, then
# `n_sympnet` gradient layers, then one `SymplecticAttentionP`, then `n_sympnet` more. With the
# defaults `L = 1` and `n_sympnet = 2` that is `1 * (2 + 2 * 2) = 6`. The unequal-dimension builder
# adds a `PSDLayer` at each end, giving 8.
#
# `init_upper` defaults to `true`, so `is_upper_criterion` is `isodd` and each run of gradient
# layers starts with a `GradientLayerQ` and alternates.

using Test
import GeometricMachineLearning: Chain, SymplecticTransformer, PSDLayer,
                                 SymplecticAttentionQ, SymplecticAttentionP,
                                 GradientLayerQ, GradientLayerP

@testset "SymplecticTransformer: the :NoUpscale Chain" begin
    model = Chain(SymplecticTransformer(4))

    @test length(model.layers) == 6

    @test model.layers[1] isa SymplecticAttentionQ
    @test model.layers[2] isa GradientLayerQ
    @test model.layers[3] isa GradientLayerP
    @test model.layers[4] isa SymplecticAttentionP
    @test model.layers[5] isa GradientLayerQ
    @test model.layers[6] isa GradientLayerP
end

@testset "SymplecticTransformer: the :Upscale Chain" begin
    model = Chain(SymplecticTransformer(4; transformer_dim = 8))

    @test length(model.layers) == 8

    # A `PSDLayer` at each end, with the six layers of the :NoUpscale chain between them.
    @test model.layers[1] isa PSDLayer
    @test model.layers[8] isa PSDLayer

    @test model.layers[2] isa SymplecticAttentionQ
    @test model.layers[5] isa SymplecticAttentionP
end
