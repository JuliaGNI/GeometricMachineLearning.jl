using GeometricMachineLearning
using Test
import Random

Random.seed!(0)

@testset "the DataLoader constructor gives the positional constructor's result" begin
    dl = DataLoader(rand(Float32, 4, 3, 5), rand(Float32, 2, 1, 5); suppress_info = true)
    @test ClassificationTransformer(dl) ===
          ClassificationTransformer(4, 2, 7, 16, softmax, true, true)
    @test ClassificationTransformer(
        dl; n_heads = 2, L = 3, activation = tanh, Stiefel = false,
        add_connection = false) ===
          ClassificationTransformer(4, 2, 2, 3, tanh, false, false)
    # The two flags differ, so a constructor that swaps them gives another architecture.
    @test ClassificationTransformer(dl; Stiefel = true, add_connection = false) ===
          ClassificationTransformer(4, 2, 7, 16, softmax, true, false)
    @test ClassificationTransformer(dl; Stiefel = false, add_connection = true) ===
          ClassificationTransformer(4, 2, 7, 16, softmax, false, true)
end
