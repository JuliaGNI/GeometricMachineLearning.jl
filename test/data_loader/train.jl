using GeometricMachineLearning
using GeometricMachineLearning: NetworkLoss, params, ZygotePullback
using AbstractNeuralNetworks: ArrayOrNamedTuple
using NeuralNetworkParameters: NetworkParameters
import ReducedComplexityModeling
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
    opt = TrainingOptimizer(nn.params; algorithm = Adam(), linesearch = T(0.01))
    history = train!(
        nn, opt, dl, Batch(8), 200; loss = FeedForwardLoss(), show_progress = false)
    @test eltype(history) === T
    @test length(history) == 200
    @test history[end] < history[1] / 100
end

# A `Chain` whose second layer is a `Chain` has a parameter set with a nested level, and the gradient
# that `ZygotePullback` returns has that level too. `Chain(::Chain, ::AbstractExplicitLayer)` flattens,
# so the inner `Chain` goes second. A full-batch `GradientMethod` step of size 0.01 on a least-squares
# loss lowers the loss, so the second entry of the history is below the first.
@testset "train! trains a Chain that holds a Chain with ZygotePullback ($T)" for T in REAL_ELTYPES
    Random.seed!(10)
    x = randn(T, 2, 20)
    dl = DataLoader(x, T[1 2; -1 3] * x; suppress_info = true)
    model = Chain(Dense(2, 3, tanh), Chain(Dense(3, 3, tanh), Dense(3, 2, identity)))
    nn = NeuralNetwork(model, T)
    @test model.layers[2] isa Chain
    @test keys(params(nn).L2) == (:L1, :L2)
    p₀ = deepcopy(params(nn))
    opt = TrainingOptimizer(nn.params; algorithm = GradientMethod(), linesearch = T(0.01))
    history = train!(nn, opt, dl, Batch(20), 2;
        loss = ZygotePullback(FeedForwardLoss()), show_progress = false)
    @test eltype(history) === T
    @test length(history) == 2
    @test history[2] < history[1]
    @test params(nn).L2.L1.W != p₀.L2.L1.W
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
        opt = TrainingOptimizer(nn.params; algorithm = Adam())
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
    @test opt₁.state.iterations == opt₂.state.iterations == 30
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
            nn, TrainingOptimizer(nn.params), dl, Batch(5), 2; loss = FeedForwardLoss(),
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
    opt = TrainingOptimizer(nn.params; algorithm = GradientMethod(), linesearch = floatmin(T))
    p₀ = deepcopy(params(nn))
    Random.seed!(8)
    losses = map(batch(dl)) do indices
        data = ReducedComplexityModeling.convert_input_and_batch_indices_to_array(
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

# Without `loss`, `train!` trains on the loss that matches the architecture and the batch, and on
# any other combination it says that it needs one.
#
# Not numeric: the elements selected are loss types, not values, so this is not run over
# `REAL_ELTYPES`.
@testset "train!'s default losses" begin
    transformer = StandardTransformerIntegrator(4; n_heads = 2, L = 1, n_blocks = 1)
    @test NetworkLoss(transformer, Batch(4, 3)) == TransformerLoss(3, 3)
    @test NetworkLoss(transformer, Batch(4, 3, 2)) == TransformerLoss(3, 2)
    # a prediction window is the batch's, not the architecture's: `TransformerLoss(batch)` takes
    # both lengths from the `Batch` it trains on
    @test_throws MethodError NetworkLoss(transformer, 1)
    @test NetworkLoss(GSympNet(4; n_layers = 2), Batch(4)) isa FeedForwardLoss
    autoencoder = SymplecticAutoencoder(10, 4)
    @test NetworkLoss(autoencoder, Batch(4)) isa AutoEncoderLoss
    # a transformer with a `Batch{:FeedForward}` and an unknown architecture both reach the fallback
    for (arch, batch) in ((transformer, Batch(4)),
        (NeuralNetwork(Chain(Dense(2, 2, identity))).architecture, Batch(4)))
        @test_throws ArgumentError NetworkLoss(arch, batch)
    end

    nn = NeuralNetwork(Chain(Dense(2, 2, identity)))
    dl = DataLoader(rand(2, 10), rand(2, 10); suppress_info = true)
    raised = try
        train!(nn, TrainingOptimizer(params(nn)), dl, Batch(5), 1; show_progress = false)
    catch e
        e
    end
    @test raised isa ArgumentError
    @test occursin("`loss`", raised.msg)

    # the same default trains on, through `train!` itself
    nn = NeuralNetwork(SymplecticAutoencoder(10, 4))
    dl = DataLoader(rand(10, 10), rand(10, 10); suppress_info = true)
    history = train!(
        nn, TrainingOptimizer(params(nn)), dl, Batch(5), 2; show_progress = false)
    @test length(history) == 2
end

# `loss` takes a `NetworkLoss` or an `AbstractPullback`. `train!` builds the pullback it trains on
# once per call and not once per step: the test-local method on `_as_pullback` counts the builds,
# and the body of the `AbstractPullback` counts the minibatches.
#
# Not numeric: the assertion counts calls, and the elements are loss types.
@testset "loss takes a pullback, and builds it once per call" begin
    Random.seed!(25)
    calls = Ref(0)
    builds = Ref(0)
    # An `AbstractPullback` whose body counts its calls: the loss it trains on is its own `loss`
    # field, so the body runs once per minibatch. The arguments are typed, because upstream's own
    # method on `AbstractPullback` is otherwise as specific in one slot and less specific in the
    # others.
    struct CountingPullback{L} <: AbstractPullback{L}
        loss::L
    end
    function (pb::CountingPullback)(
            ps::NetworkParameters, model,
            input_nt_output_nt::Tuple{<:ArrayOrNamedTuple, <:ArrayOrNamedTuple})
        calls[] += 1
        ZygotePullback(pb.loss)(ps, model, input_nt_output_nt)
    end
    # A `NetworkLoss` whose `_as_pullback` is a method of this file: `train!` calls `_as_pullback`
    # once per call, so the count is of its builds and not of its steps.
    struct CountingLoss <: NetworkLoss end
    function GeometricMachineLearning._as_pullback(::CountingLoss)
        builds[] += 1
        ZygotePullback(FeedForwardLoss())
    end

    nn = NeuralNetwork(Chain(Dense(2, 2, identity)))
    dl = DataLoader(rand(2, 12), rand(2, 12); suppress_info = true)
    train!(nn, TrainingOptimizer(params(nn)), dl, Batch(4), 1;
        loss = CountingPullback(FeedForwardLoss()), show_progress = false)
    @test calls[] == 3

    train!(nn, TrainingOptimizer(params(nn)), dl, Batch(4), 1;
        loss = CountingLoss(), show_progress = false)
    @test builds[] == 1

    # a keyword that names a pullback of `loss` is not one of `train!`'s
    @test_throws MethodError train!(nn, TrainingOptimizer(params(nn)), dl, Batch(4), 1;
        loss = FeedForwardLoss(), pullback = ZygotePullback(FeedForwardLoss()),
        show_progress = false)
    # and a loss that is neither a `NetworkLoss` nor an `AbstractPullback`
    @test_throws TypeError train!(nn, TrainingOptimizer(params(nn)), dl, Batch(4), 1;
        loss = 1, show_progress = false)
end

@testset "train! refuses n_epochs < 1 before any step" begin
    Random.seed!(5)
    nn = NeuralNetwork(Chain(Dense(2, 2, identity)))
    dl = DataLoader(rand(2, 10), rand(2, 10); suppress_info = true)
    opt = TrainingOptimizer(nn.params)
    p₀ = deepcopy(params(nn))
    for n in (0, -1)
        @test_throws ArgumentError train!(
            nn, opt, dl, Batch(5), n; loss = FeedForwardLoss(),
            show_progress = false)
    end
    @test opt.state.iterations == 0
    @test params(nn) == p₀
end
