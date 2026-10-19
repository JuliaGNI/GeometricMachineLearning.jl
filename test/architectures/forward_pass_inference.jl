# Every architecture the package exports infers the element type of its forward pass. `@inferred` is
# the assertion: a consolidation that turned a statically dispatched call into a dynamic one would
# return the right numbers and hide the `Any` downstream, which no value assertion catches.
#
# The list holds every exported architecture that can be applied to data: the two SympNets, the
# volume-preserving feedforward network and transformer, the ResNet, the two compressions, the four
# transformer integrators, the generic `Transformer`, the `ClassificationTransformer`, and the
# Hamiltonian and Lagrangian networks. `GeneralizedHamiltonianArchitecture`'s constructor throws and
# builds no network, so it has no forward pass to check. The structured leaves a chain of them can
# hold are covered by a `Stiefel` transformer (a `StiefelManifold` weight) and a `skew_sym`
# volume-preserving transformer (a `SkewSymMatrix` weight), since both defaults select the dense
# leaf.

using GeometricMachineLearning
using Test
import Random

GML = GeometricMachineLearning

Random.seed!(1234)

# Each architecture with data of a shape its chain accepts. The data are built here, so that the
# assertion below sees the forward pass and nothing else.
function cases(::Type{T}) where {T}
    (GSympNet(4; n_layers = 2), rand(T, 4)),
    (LASympNet(4; depth = 2, nhidden = 1), rand(T, 4)),
    (VolumePreservingFeedForward(4, 1), rand(T, 4)),
    (ResNet(4, 1, tanh), rand(T, 4)),
    (SymplecticAutoencoder(6, 4; n_encoder_blocks = 2, n_decoder_blocks = 2), rand(T, 6)),
    (PSDArch(6, 4), rand(T, 6)),
    (StandardTransformerIntegrator(4; n_heads = 2, L = 1), rand(T, 4, 3)),
    (LinearSymplecticTransformer(4, 3; L = 1), rand(T, 4, 3)),
    (SymplecticTransformer(4; L = 1), rand(T, 4, 3)),
    (VolumePreservingTransformer(4, 3; L = 1), rand(T, 4, 3)),
    (VolumePreservingTransformer(4, 3; L = 1, skew_sym = true), rand(T, 4, 3)),
    (Chain(Transformer(4, 2, 1)), rand(T, 4, 3)),
    (Chain(Transformer(4, 2, 1; Stiefel = true)), rand(T, 4, 3)),
    (ClassificationTransformer(4, 3, 2, 1, softmax, true, true), rand(T, 4, 3, 2)),
    (ClassificationTransformer(4, 3, 2, 1, softmax, false, true), rand(T, 4, 3, 2)),
    (StandardHamiltonianArchitecture(2, 4), rand(T, 2)),
    (LagrangianNeuralNetwork(2; width = 4), rand(T, 2))
end

# A function barrier: `arch` and `x` arrive with concrete types, so `@inferred` judges the forward
# pass rather than the loop's `Union`.
function assert_forward_infers(arch, x::AbstractArray{T}) where {T}
    nn = NeuralNetwork(arch, T)
    y = @inferred nn(x)
    @test y isa AbstractArray{T}
    @test eltype(y) === T
    y
end

@testset "the forward pass of every exported architecture infers its element type ($T)" for T in (
    Float32, Float64)
    for (arch, x) in cases(T)
        @testset "$(nameof(typeof(arch)))" begin
            assert_forward_infers(arch, x)
        end
    end
end
