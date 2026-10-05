using GeometricMachineLearning, Test, KernelAbstractions
import GeometricOptimizers
import Random

# A network with a Stiefel layer beside a Euclidean one is the shape `CompositeMethod` exists for:
# `ScalarMomentAdam` steps a single `StiefelManifold`, so the dense layer needs another method. The
# per-leaf caches and states are `GeometricOptimizers`'; what is pinned here is that `Optimizer`
# hands a composite through to them unchanged.

function stiefel_beside_dense(T)
    Random.seed!(7)
    NeuralNetwork(Chain(StiefelLayer(6, 3), Dense(6, 6, tanh)), KernelAbstractions.CPU(), T)
end

function minibatch_gradient(T, k)
    Random.seed!(100 + k)
    (L1 = (weight = randn(T, 6, 3),), L2 = (W = randn(T, 6, 6), b = randn(T, 6)))
end

@testset "a composite gives every leaf the state of the method it selects" begin
    nn = stiefel_beside_dense(Float32)
    opt = Optimizer(CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam()), nn)
    states = opt.training.state.states
    @test opt.training.state isa CompositeState{Float32}
    @test states.L1.weight isa ScalarMomentAdamState{Float32}
    @test states.L2.W isa AdamState{Float32}
    @test states.L2.b isa AdamState{Float32}

    ps = GeometricMachineLearning.params(nn)
    for k in 1:3
        optimization_step!(ps, opt, minibatch_gradient(Float32, k))
    end
    @test ps.L1.weight isa StiefelManifold
    @test GeometricOptimizers.check(ps.L1.weight) < 1.0f-5
    @test GeometricOptimizers.iteration_number(opt.training.state) == 3
end

@testset "a composite of one method is that method, exactly" begin
    # Each optimizer is built straight after its network, so both draw their section completions
    # from the same position of the seed.
    whole = stiefel_beside_dense(Float64)
    whole_opt = Optimizer(Adam(), whole)
    per_leaf = stiefel_beside_dense(Float64)
    per_leaf_opt = Optimizer(CompositeMethod(; manifold = Adam(), array = Adam()), per_leaf)
    for k in 1:5
        optimization_step!(GeometricMachineLearning.params(whole), whole_opt,
            minibatch_gradient(Float64, k))
        optimization_step!(GeometricMachineLearning.params(per_leaf), per_leaf_opt,
            minibatch_gradient(Float64, k))
    end
    a, b = GeometricMachineLearning.params(whole), GeometricMachineLearning.params(per_leaf)
    @test a.L1.weight.A == b.L1.weight.A
    @test a.L2.W == b.L2.W
    @test a.L2.b == b.L2.b
end
