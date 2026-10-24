# The forward pass of every architecture that the package exports infers its result type, and the
# result has the element type of the network.

using GeometricMachineLearning
using Test
import Random

include("../helpers/eltypes.jl")

input(::Type{T}, dim, cols) where {T} = T[sin(i + 2j) for i in 1:dim, j in 1:cols]

# `(name, architecture, input dimension, number of columns)`.
const ARCHITECTURES = (
    ("ResNet", ResNet(4, 2, tanh), 4, 3),
    ("StandardTransformerIntegrator", StandardTransformerIntegrator(4; n_heads = 2), 4, 3),
    # The default is one head per dimension, so this is a 32-head attention.
    ("StandardTransformerIntegrator, default heads",
        StandardTransformerIntegrator(32), 32, 3),
    ("VolumePreservingTransformer", VolumePreservingTransformer(4, 3), 4, 3),
    ("LASympNet", LASympNet(4), 4, 3),
    ("GSympNet", GSympNet(4), 4, 3),
    ("ClassificationTransformer",
        ClassificationTransformer(4, 2, 2, 1, softmax, false, true), 4, 3),
    ("VolumePreservingFeedForward", VolumePreservingFeedForward(4), 4, 3),
    ("SymplecticAutoencoder", SymplecticAutoencoder(8, 4), 8, 3),
    ("PSDArch", PSDArch(8, 4), 8, 3),
    ("LinearSymplecticTransformer", LinearSymplecticTransformer(4, 3), 4, 3),
    ("SymplecticTransformer", SymplecticTransformer(4), 4, 3))

@testset "the forward pass of $name infers ($T)" for (name, arch, dim, cols) in ARCHITECTURES,
    T in REAL_ELTYPES

    Random.seed!(0)
    nn = NeuralNetwork(arch, CPU(), T)
    x = input(T, dim, cols)
    y = @inferred nn(x)
    @test eltype(y) === T
end
