# Both `compute_output_of_mha` methods concatenate the head outputs in one call. The matrix method
# reaches Base's linear `vcat` through `reduce`; the tensor method cannot, because `reduce(vcat, …)`
# folds pairwise on a 3-tensor, so it splats -- and a splatted vector hides the argument count from
# inference, which is why that call carries a result-type assertion. Nothing else in `test/` pins
# the inference of a forward pass, so an inference regression in either method is invisible to the
# suite: the layer still returns the right numbers, and the `Any` propagates out through the functor
# into everything downstream. See `CHANGELOG.md` for the change this guards.

using GeometricMachineLearning
using Test
import Random, KernelAbstractions

GML = GeometricMachineLearning

Random.seed!(1234)

mha_parameters(d, T) = GML.params(NeuralNetwork(
    Chain(d), KernelAbstractions.CPU(), T))[1]

@testset "compute_output_of_mha infers concretely, Stiefel = $Stiefel" for Stiefel in (false, true)
    dim, n_heads, seq_length, n_data = 8, 4, 6, 3
    d = MultiHeadAttention(dim, n_heads; Stiefel = Stiefel)
    ps = mha_parameters(d, Float32)

    x_matrix = rand(Float32, dim, seq_length)
    x_tensor = rand(Float32, dim, seq_length, n_data)

    @test (@inferred GML.compute_output_of_mha(d, x_matrix, ps)) isa Matrix{Float32}
    @test (@inferred GML.compute_output_of_mha(d, x_tensor, ps)) isa Array{Float32, 3}

    # The functor is what the rest of the network calls, so it is the type that actually propagates.
    @test (@inferred d(x_matrix, ps)) isa Matrix{Float32}
    @test (@inferred d(x_tensor, ps)) isa Array{Float32, 3}
end
