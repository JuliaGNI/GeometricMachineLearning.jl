using GeometricMachineLearning
using Test
import Random

Random.seed!(123)

# A network with a Stiefel leaf trains on data of the other precision, as a network of `Dense`
# layers does: the products promote, and every leaf keeps the network's element type.
@testset "a Stiefel network trains on data of the other precision" begin
    for (T, S) in ((Float64, Float32), (Float32, Float64))
        x = rand(S, 6, 12)
        nn = NeuralNetwork(Chain(StiefelLayer(6, 3), Dense(3, 2, tanh)), T)
        before = deepcopy(nn.params)
        train!(nn, TrainingOptimizer(nn.params), DataLoader(x, x[1:2, :]), Batch(4), 2;
            loss = FeedForwardLoss())
        leaves = (values(nn.params.L1)..., values(nn.params.L2)...)
        @test all(leaf -> eltype(leaf) === T, leaves)
        @test nn.params.L1.weight != before.L1.weight
    end
end
