using GeometricMachineLearning
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(123)

# compare the `init_upper=true` to `init_upper=false`
function compare_init_upper_with_init_lower(
        dim::Integer, seq_length::Integer, third_dim::Integer = 10; T::Type = Float64)
    arch₁ = LinearSymplecticTransformer(dim, seq_length; init_upper = true)
    arch₂ = LinearSymplecticTransformer(dim, seq_length; init_upper = false)

    nn₁ = NeuralNetwork(arch₁, T)

    model₂ = Chain(arch₂)
    nn₂ = NeuralNetwork(arch₂, model₂, nn₁.params, nn₁.backend)

    test_matrix = rand(T, dim, seq_length)
    test_tensor = rand(T, dim, seq_length, third_dim)

    @test eltype(nn₁(test_matrix)) === T
    @test eltype(nn₁(test_tensor)) === T
    @test nn₁(test_matrix) ≉ nn₂(test_matrix)
    @test nn₁(test_tensor) ≉ nn₂(test_tensor)
end

for T in REAL_ELTYPES
    compare_init_upper_with_init_lower(5, 4; T)
    compare_init_upper_with_init_lower(3, 10; T)
end
