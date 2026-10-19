# The activation keyword of `LinearSymplecticTransformer` reaches its chain, and the upscaling
# `Dense` layer of `StandardTransformerIntegrator` takes the activation its keyword names.

using GeometricMachineLearning
using AbstractNeuralNetworks: Activation
using Test

GML = GeometricMachineLearning

@testset "LinearSymplecticTransformer keeps the activation it is given" begin
    arch = LinearSymplecticTransformer(4, 3; activation = sin)
    @test arch.activation === sin
    @test typeof(arch).parameters[1] === typeof(sin)

    # Every gradient layer of the chain carries it, which is what the constructor's type parameter
    # is read by.
    gradient_layers = filter(l -> l isa GML.GradientLayer, Chain(arch).layers)
    @test !isempty(gradient_layers)
    @test all(l -> l.activation === sin, gradient_layers)

    # The default is still the constant the docstring interpolates.
    @test LinearSymplecticTransformer(4, 3).activation ===
          GeometricMachineLearning.lst_activation_default
end

@testset "StandardTransformerIntegrator names its upscaling activation" begin
    @test hasfield(StandardTransformerIntegrator, :upscaling_activation)
    @test !hasfield(StandardTransformerIntegrator, :upsacling_activation)

    arch = StandardTransformerIntegrator(4; transformer_dim = 6, n_heads = 2,
        upscaling_activation = tanh)
    @test arch.upscaling_activation === tanh

    # The upscaling `Dense` is the first layer of the chain, and it carries the activation. The
    # `Dense` stores `Activation(tanh)`, so that is what "carries `tanh`" means here.
    layers = Chain(arch).layers
    @test layers[1] isa Dense
    @test layers[1].σ === Activation(tanh)

    # The output `Dense` scales back to the system dimension with `identity`, unconditionally.
    @test layers[end] isa Dense
    @test layers[end].σ === Activation(identity)

    # The default is the documented constant, and it is the one a non-upscaling chain never reads.
    @test StandardTransformerIntegrator(4; transformer_dim = 6, n_heads = 2).upscaling_activation ===
          GeometricMachineLearning.sti_upscaling_activation_default

    # A chain that does not change the dimension has no upscaling layer, and the keyword is not
    # read: the first layer is the attention layer.
    no_upscale = Chain(StandardTransformerIntegrator(4; n_heads = 2, L = 1)).layers
    @test no_upscale[1] isa MultiHeadAttention
end
