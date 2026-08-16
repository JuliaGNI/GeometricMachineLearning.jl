using GeometricMachineLearning
using GeometricMachineLearning: nruns, opt, method, batchsize
using Test

include("data/data_generation.jl")

# `TrainingParameters`' third argument used to default to `default_optimizer()`, which was deleted
# along with the rest of GML's optimizer layer when it moved to GeometricOptimizers. The default was
# left behind, so `TrainingParameters(nruns, method)` raised
# `UndefVarError: default_optimizer not defined` for anyone who used it.
#
# Nothing caught that: the only tests that touch `TrainingParameters` live under `test/train!/`,
# which `runtests.jl` does not include, and they all pass the optimizer explicitly anyway. Hence this
# file, which *is* included.
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

# The step size moved from the optimizer method onto `Optimizer`, and `train!` built its `Optimizer`
# without forwarding one — so the step size could not be set through `train!` at all.
@testset "train! forwards a step size" begin
    nn = NeuralNetwork(GSympNet(2; n_layers = 2), Float64)
    m = BasicSympNet()
    o = GradientOptimizer()

    loss_default = train!(nn, tra_ps_data, o, m; ntraining = 1, batch_size = (1, 2))
    @test length(loss_default) == 1

    nn2 = NeuralNetwork(GSympNet(2; n_layers = 2), Float64)
    loss_small = train!(nn2, tra_ps_data, o, m; ntraining = 1, batch_size = (1, 2), step_size = 1e-8)
    @test length(loss_small) == 1
end
