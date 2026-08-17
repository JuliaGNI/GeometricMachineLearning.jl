using GeometricMachineLearning
using GeometricMachineLearning: nruns, opt, method, batchsize
using Test

include("data/data_generation.jl")

# This file covers the `TrainingParameters` constructors and `train!`'s keywords. It is the only
# coverage of either that `runtests.jl` includes: the rest lives under `test/train!/`, which
# `runtests.jl` does not run.
#
# `TrainingParameters` takes the optimizer as a required third argument — there is no default
# optimizer — which is what the testset below pins, along with the accessors and the keyword copy
# constructor.
@testset "TrainingParameters takes the optimizer explicitly" begin
    m = BasicSympNet()
    o = GradientOptimizer()

    tp = TrainingParameters(3, m, o)
    @test tp isa TrainingParameters
    @test nruns(tp) == 3
    @test opt(tp) === o
    @test method(tp) === m
    @test batchsize(tp) === missing

    tp_bs = TrainingParameters(3, m, o; batch_size = (1, 2))
    @test batchsize(tp_bs) == (1, 2)

    # the copy constructor has to keep working
    @test opt(TrainingParameters(tp; nruns = 5)) === o
    @test nruns(TrainingParameters(tp; nruns = 5)) == 5
end

# The step size is a property of the `Optimizer`, not of the optimization method, so `train!` has
# to forward its `step_size` keyword when it builds one. A `train!` that does not forward it takes
# `_default_step_size(m)` on every run and trains regardless of what the caller asked for, which is
# silent rather than an error — hence a test that observes whether the network moves.
#
# `train!` recomputes the loss over the *whole* data set after every step, so the value it stores
# does not depend on which batch was drawn. That gives an assertion needing no seed: at
# `step_size = 0` no parameter can move, hence every entry of the loss array is the same number.
@testset "train! forwards a step size" begin
    m = BasicSympNet()
    o = GradientOptimizer()
    ntraining = 5

    nn_frozen = NeuralNetwork(GSympNet(2; n_layers = 2), Float64)
    loss_frozen = train!(nn_frozen, tra_ps_data, o, m; ntraining, batch_size = (1, 2), step_size = 0.0)
    @test length(loss_frozen) == ntraining
    @test all(loss_frozen .== loss_frozen[1])

    nn_moving = NeuralNetwork(GSympNet(2; n_layers = 2), Float64)
    loss_moving = train!(nn_moving, tra_ps_data, o, m; ntraining, batch_size = (1, 2), step_size = 1e-2)
    @test length(loss_moving) == ntraining
    @test !all(loss_moving .== loss_moving[1])
end
