# The reproduction script `scripts/reproduction/symplectic_autoencoders/training.jl` has a `solve!`
# of a PSD architecture on a snapshot matrix, and `solve!` takes a `:RegularData` loader. That is
# why its `DataLoader(data)` passes `autoencoder = true`, and why a matrix that does not is a
# `MethodError` rather than a training run on input-shifted targets.

using GeometricMachineLearning
using GeometricMachineLearning: solve!
using Test
import Random

Random.seed!(1234)

@testset "solve! of a PSD architecture needs a samples loader" begin
    data = rand(Float32, 6, 20)
    nn = NeuralNetwork(PSDArch(6, 2), CPU(), Float32)
    @test isfinite(solve!(nn, DataLoader(data; autoencoder = true, suppress_info = true)))
    @test_throws MethodError solve!(nn, DataLoader(data; suppress_info = true))
end
