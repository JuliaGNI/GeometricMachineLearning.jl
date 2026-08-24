using GeometricMachineLearning, Test, LinearAlgebra, KernelAbstractions
using AbstractNeuralNetworks: AbstractExplicitLayer
import GeometricOptimizers
import GeometricMachineLearning: NeuralNetwork
import Random

Random.seed!(1234)

function optimization_step_test(N, n, T)
    model = Chain(StiefelLayer(N, n), Dense(N, N, tanh))
    ps = NeuralNetwork(model, KernelAbstractions.CPU(), T).params
    # gradient 
    dx = (L1 = (weight=rand(Float32, N, n),), L2 = (W=rand(Float32, N, N), b=rand(Float32, N)))
    m = AdamOptimizer()
    # randomize the cache!
    o = Optimizer(m, ps)

    ps2 = deepcopy(ps)
    λY = GlobalSection(ps)
    optimization_step!(o, λY, ps, dx)
    @test typeof(ps[1].weight) <: StiefelManifold
    for (layers1, layers2) in zip(values(ps), values(ps2))
        for key in keys(layers1)
            @test norm(layers1[key] - layers2[key]) > T(1f-6)
        end
    end
end

N_max = 10
T = Float32
for N = 4:N_max
    for n = 1:N
        optimization_step_test(N, n, T)
    end
end

# The regression net for the branch order in `_make_optimizer_cache`/`_make_optimizer_state`: a
# `NetworkParameters` is a tree to descend into, so it is recognised *structurally*, before the
# capability question `_use_go_cache` asks. The shape that follows is one `GeometricOptimizers` cache
# per *layer* -- not one for the root, which is what putting the capability question first would give
# the moment `GeometricOptimizers` adds the container to `OptimizerSolution`, and not one per weight,
# which is what hoisting the `NamedTuple` branch as well would give.
@testset "one cache and one state per layer, keyed by the network's layers" begin
    model = Chain(StiefelLayer(6, 3), Dense(6, 6, tanh))
    nn = NeuralNetwork(model, KernelAbstractions.CPU(), Float32)
    o = Optimizer(AdamOptimizer(), nn)

    for tree in (o.cache, o.state)
        @test tree isa NamedTuple
        @test keys(tree) == keys(GeometricMachineLearning.params(nn))
    end
    @test o.cache.L1 isa GeometricOptimizers.OptimizerCache
    @test o.cache.L2 isa GeometricOptimizers.OptimizerCache
    @test o.state.L1 isa GeometricOptimizers.OptimizerState
    @test o.state.L2 isa GeometricOptimizers.OptimizerState
end
