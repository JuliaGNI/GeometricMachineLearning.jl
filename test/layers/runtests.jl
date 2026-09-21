using SafeTestsets

@safetestset "Gradient Layer" begin
    include("gradient_layer_tests.jl")
end
@safetestset "Test tensor-slice consistency of sympnet layers" begin
    include("sympnet_layers_test.jl")
end
@safetestset "Test symplecticity of the sympnet upscaling layer" begin
    include("sympnet_upscaling.jl")
end
@safetestset "Manifold Neural Network Layers" begin
    include("manifold_layers.jl")
end
@safetestset "The manifold layers initialise an orthonormal weight" begin
    include("manifold_layer_orthonormality.jl")
end
@safetestset "ResNet" begin
    include("resnet_tests.jl")
end
@safetestset "Classification layer" begin
    include("classification.jl")
end
@safetestset "Test volume-preserving feedforward neural network" begin
    include("volume_preserving_feedforward.jl")
end
@safetestset "parameterlength(::PSDLayer) is exact" begin
    include("psd_parameterlength.jl")
end
@safetestset "parameterlength(::GrassmannLayer) is an integer count" begin
    include("grassmann_parameterlength.jl")
end
