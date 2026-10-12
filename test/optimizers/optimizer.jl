using GeometricMachineLearning
using GeometricMachineLearning: params, convert_input_and_batch_indices_to_array
using Test
import Random
import Zygote

include("../helpers/eltypes.jl")

# One epoch of one minibatch is one gradient step at `p₀`, so the result is `p₀ - η∇L(p₀)` with the
# `algorithm` and the `linesearch` the constructor was given. The reference gradient is taken on the
# minibatch `train!` draws, drawn again from the same seed. The bound is 4 eps(T), for the rounding of
# one step.
@testset "NetworkOptimizer takes algorithm and linesearch ($T)" for T in REAL_ELTYPES
    Random.seed!(1)
    x, y = randn(T, 2, 8), randn(T, 2, 8)
    dl = DataLoader(x, y; suppress_info = true)
    model = Chain(Dense(2, 2, identity))
    nn = NeuralNetwork(model, T)
    p₀ = deepcopy(params(nn))
    η = T(0.3)

    Random.seed!(2)
    batch = Batch(8)
    minibatches = batch(dl)
    @test length(minibatches) == 1
    input, output = convert_input_and_batch_indices_to_array(dl, batch, only(minibatches))
    g = only(Zygote.gradient(ps -> FeedForwardLoss()(model, ps, input, output), p₀))

    opt = NetworkOptimizer(nn; algorithm = GradientMethod(), linesearch = η)
    Random.seed!(2)
    train!(nn, opt, dl, batch, 1; loss = FeedForwardLoss(), show_progress = false)
    for k in (:W, :b)
        @test params(nn).L1[k] isa Array{T}
        @test params(nn).L1[k] ≈ p₀.L1[k] .- η .* g.L1[k] rtol = 4eps(T)
        @test params(nn).L1[k] != p₀.L1[k]
    end

    # the same run with the optimizer built from the parameter set
    nn₂ = NeuralNetwork(model, model, deepcopy(p₀), CPU())
    opt₂ = NetworkOptimizer(params(nn₂); algorithm = GradientMethod(), linesearch = η)
    Random.seed!(2)
    train!(nn₂, opt₂, dl, batch, 1; loss = FeedForwardLoss(), show_progress = false)
    @test params(nn₂) == params(nn)
end

@testset "AdamOptimizerWithDecay splats into the keywords" begin
    nn = NeuralNetwork(Chain(Dense(2, 2, identity)), Float32)
    opt = NetworkOptimizer(nn; AdamOptimizerWithDecay(100)...)
    @test opt.optimizer.method isa Adam{Float32}
    @test opt.optimizer.linesearch isa DecayingStatic{Float32}
end
