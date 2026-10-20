# The forward pass of every architecture that GeometricMachineLearning exports infers its result
# type, in each real element type. `GeneralizedHamiltonianArchitecture` is left out: its constructor
# raises, so there is no network to apply.

using GeometricMachineLearning
using Test
import Random

include("../helpers/eltypes.jl")

const ARCHITECTURES = [
    ("GSympNet", GSympNet(4), (4, 3)),
    ("LASympNet", LASympNet(4), (4, 3)),
    ("LinearSymplecticTransformer", LinearSymplecticTransformer(4, 3), (4, 3)),
    ("SymplecticTransformer", SymplecticTransformer(4; symmetric = false), (4, 3)),
    ("SymplecticTransformer upscaled",
        SymplecticTransformer(4; transformer_dim = 6), (4, 3)),
    ("StandardTransformerIntegrator",
        StandardTransformerIntegrator(4; n_heads = 2), (4, 3)),
    ("StandardTransformerIntegrator on a tensor",
        StandardTransformerIntegrator(4; n_heads = 2), (4, 3, 2)),
    ("VolumePreservingTransformer",
        VolumePreservingTransformer(3, 3; skew_sym = true), (3, 3)),
    ("VolumePreservingFeedForward",
        VolumePreservingFeedForward(3; init_upper = true), (3, 3)),
    ("SymplecticAutoencoder", SymplecticAutoencoder(8, 4; encoder_init_q = false), (8, 2)),
    ("PSDArch", PSDArch(8, 4), (8, 2)),
    ("ResNet", ResNet(4, 2, tanh), (4, 3)),
    ("ClassificationTransformer",
        ClassificationTransformer(4, 3, 2, 1, softmax, true, true), (4, 3, 2)),
    ("StandardHamiltonianArchitecture", StandardHamiltonianArchitecture(4), (4, 3)),
    ("LagrangianNeuralNetwork", LagrangianNeuralNetwork(4), (4, 3))
]

@testset "the forward pass of $name infers ($T)" for (name, arch, dims) in ARCHITECTURES,
    T in REAL_ELTYPES

    nn = NeuralNetwork(arch, CPU(), T; rng = Random.Xoshiro(0))
    x = rand(Random.Xoshiro(1), T, dims...)
    y = @inferred nn(x)
    @test eltype(y) === T
end
