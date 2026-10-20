# `GSympNet`, `LinearSymplecticTransformer` and `SymplecticTransformer` build their gradient layers
# with one function, which takes the dimension and the activation as arguments rather than reading
# them off the architecture. `SymplecticTransformer` passes its `transformer_dim`, so with an
# upscaling its gradient layers have the upscaled dimension.

using GeometricMachineLearning
using GeometricMachineLearning: GradientLayer
using Test

const GML = GeometricMachineLearning

@testset "one builder for the gradient layers" begin
    @test length(methods(GML._gradient_layers)) == 1
    @test !isdefined(GML, :_make_block_for_initialization)
end

gradient_layers(arch) = filter(l -> l isa GradientLayer, Chain(arch).layers)

@testset "each architecture passes its own dimension and activation" begin
    for (arch, dim, activation) in (
        (GSympNet(4; n_layers = 3, activation = sin), 4, sin),
        (LinearSymplecticTransformer(4, 3; activation = sin), 4, sin),
        (SymplecticTransformer(4; sympnet_activation = sin), 4, sin),
        (SymplecticTransformer(4; transformer_dim = 6, sympnet_activation = sin), 6, sin))
        layers = gradient_layers(arch)
        @test !isempty(layers)
        @test all(l -> l isa GradientLayer{dim, dim}, layers)
        @test all(l -> l.activation === activation, layers)
    end
end
