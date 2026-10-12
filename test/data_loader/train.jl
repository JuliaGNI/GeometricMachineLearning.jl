using GeometricMachineLearning
using GeometricMachineLearning: params
using Test
import Random

include("../helpers/eltypes.jl")

# The target is in the model class, so the loss goes to zero; the factor 100 leaves margin in
# `Float32`.
@testset "train! fits a linear map ($T)" for T in REAL_ELTYPES
    Random.seed!(1)
    A = T[1 2; -1 3]
    x = randn(T, 2, 40)
    dl = DataLoader(x, A * x; suppress_info = true)
    nn = NeuralNetwork(Chain(Dense(2, 2, identity)), T)
    opt = NetworkOptimizer(nn; algorithm = Adam(), linesearch = T(0.01))
    history = train!(
        nn, opt, dl, Batch(8), 200; loss = FeedForwardLoss(), show_progress = false)
    @test eltype(history) === T
    @test length(history) == 200
    @test history[end] < history[1] / 100
end

# The iteration number, the moments and the global sections live in `opt` only, so two calls of 5
# epochs are one call of 10. Seeding before the network and the optimizer makes the minibatches and
# the random completions of the sections the same draws in both runs.
@testset "a second call of train! resumes the first ($T)" for T in REAL_ELTYPES
    x = randn(Random.Xoshiro(2), T, 6, 30)
    dl = DataLoader(x, x[1:2, :]; suppress_info = true)
    function trained(epochs...)
        Random.seed!(3)
        nn = NeuralNetwork(Chain(StiefelLayer(6, 3), Dense(3, 2, tanh)), T)
        opt = NetworkOptimizer(nn; algorithm = Adam())
        for n in epochs
            train!(
                nn, opt, dl, Batch(10), n; loss = FeedForwardLoss(), show_progress = false)
        end
        nn, opt
    end
    nn₁, opt₁ = trained(5, 5)
    nn₂, opt₂ = trained(10)
    @test params(nn₁).L1.weight isa StiefelManifold{T}
    @test params(nn₁) == params(nn₂)
    @test opt₁.optimizer.state.iterations == opt₂.optimizer.state.iterations == 30
end

# The history has the type of the loss value, and nothing is converted: the network's element type
# wins over the data's.
@testset "the history has the element type of the loss" begin
    for (Tnn, Tdata, Thistory) in ((Float64, Float32, Float64), (Float32, Float32, Float32),
        (Float32, Int, Float64))
        Random.seed!(4)
        x = Tdata <: Integer ? rand(Tdata(-3):Tdata(3), 2, 10) : rand(Tdata, 2, 10)
        dl = DataLoader(x, x; suppress_info = true)
        nn = NeuralNetwork(Chain(Dense(2, 2, identity)), Tnn)
        history = train!(
            nn, NetworkOptimizer(nn), dl, Batch(5), 2; loss = FeedForwardLoss(),
            show_progress = false)
        @test eltype(history) === Thistory
    end
end

# A step of size `floatmin(T)` rounds away against parameters of order one, so every minibatch loss is
# the loss at `p₀`, and entry 1 of the history is the mean of three numbers that are computed here
# from the same minibatches.
@testset "the history holds the mean of the minibatch losses ($T)" for T in REAL_ELTYPES
    dl = DataLoader(randn(Random.Xoshiro(6), T, 2, 12), randn(Random.Xoshiro(7), T, 2, 12);
        suppress_info = true)
    batch = Batch(4)
    loss = FeedForwardLoss()
    nn = NeuralNetwork(Chain(Dense(2, 2, tanh)), T)
    opt = NetworkOptimizer(nn; algorithm = GradientMethod(), linesearch = floatmin(T))
    p₀ = deepcopy(params(nn))
    Random.seed!(8)
    losses = map(batch(dl)) do indices
        data = GeometricMachineLearning.convert_input_and_batch_indices_to_array(
            dl, batch, indices)
        loss(nn.model, params(nn), data...)
    end
    Random.seed!(8)
    history = train!(nn, opt, dl, batch, 1; loss, show_progress = false)
    @test params(nn) == p₀
    @test eltype(losses) === eltype(history) === T
    @test length(losses) == 3
    @test !(losses[3] ≈ sum(losses) / 3)
    @test history[1] ≈ sum(losses) / 3
end

# Without `loss`, `train!` trains on the loss that matches the architecture and the batch.
@testset "train!'s default losses" begin
    transformer_batch = Batch(4, 3)
    @test GeometricMachineLearning._default_loss(GSympNet(4; n_layers = 2), Batch(4)) isa
          FeedForwardLoss
    @test GeometricMachineLearning._default_loss(
        StandardTransformerIntegrator(4; n_heads = 2, L = 1, n_blocks = 1),
        transformer_batch) == TransformerLoss(transformer_batch)
    autoencoder = SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)
    @test GeometricMachineLearning._default_loss(autoencoder, Batch(4)) isa AutoEncoderLoss
end

@testset "train! refuses n_epochs < 1 before any step" begin
    Random.seed!(5)
    nn = NeuralNetwork(Chain(Dense(2, 2, identity)))
    dl = DataLoader(rand(2, 10), rand(2, 10); suppress_info = true)
    opt = NetworkOptimizer(nn)
    p₀ = deepcopy(params(nn))
    for n in (0, -1)
        @test_throws ArgumentError train!(
            nn, opt, dl, Batch(5), n; loss = FeedForwardLoss(),
            show_progress = false)
    end
    @test opt.optimizer.state.iterations == 0
    @test params(nn) == p₀
end
