# Both `compute_output_of_mha` methods concatenate the head outputs in one call. The matrix method
# reaches Base's linear `vcat` through `reduce`; the tensor method cannot, because `reduce(vcat, …)`
# folds pairwise on a 3-tensor, so it splats -- and a splatted vector hides the argument count from
# inference, which is why that call carries a result-type assertion. An inference regression in
# either method returns the right numbers, and the `Any` propagates out through the functor into
# everything downstream, so this file pins the inference of both methods, at 4 heads and at 32: a
# `map` over a tuple of more than 31 heads no longer infers. `test/architectures/forward_inference.jl`
# pins the forward pass of each architecture.

using GeometricMachineLearning
using Test
import Random, KernelAbstractions

include("../helpers/eltypes.jl")

GML = GeometricMachineLearning

Random.seed!(1234)

mha_parameters(d, T) = GML.params(NeuralNetwork(
    Chain(d), KernelAbstractions.CPU(), T))[1]

@testset "compute_output_of_mha infers concretely, $n_heads heads, Stiefel = $Stiefel, $T" for n_heads in (
        4, 32),
    Stiefel in (false, true),
    T in REAL_ELTYPES
    dim, seq_length, n_data = 2n_heads, 6, 3
    d = MultiHeadAttention(dim, n_heads; Stiefel = Stiefel)
    ps = mha_parameters(d, T)

    x_matrix = rand(T, dim, seq_length)
    x_tensor = rand(T, dim, seq_length, n_data)

    @test (@inferred GML.compute_output_of_mha(d, x_matrix, ps)) isa Matrix{T}
    @test (@inferred GML.compute_output_of_mha(d, x_tensor, ps)) isa Array{T, 3}
    @test eltype(GML.compute_output_of_mha(d, x_matrix, ps)) === T

    # The functor is what the rest of the network calls, so it is the type that actually propagates.
    @test (@inferred d(x_matrix, ps)) isa Matrix{T}
    @test (@inferred d(x_tensor, ps)) isa Array{T, 3}
end
