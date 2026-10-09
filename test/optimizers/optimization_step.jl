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
    dx = (L1 = (weight = rand(Float32, N, n),),
        L2 = (W = rand(Float32, N, N), b = rand(Float32, N)))
    m = AdamOptimizer()
    # randomize the cache!
    o = Optimizer(m, ps)

    ps2 = deepcopy(ps)
    optimization_step!(ps, o, dx)
    @test typeof(ps[1].weight) <: StiefelManifold
    for (layers1, layers2) in zip(values(ps), values(ps2))
        for key in keys(layers1)
            @test norm(layers1[key] - layers2[key]) > T(1.0f-6)
        end
    end
end

N_max = 10
T = Float32
for N in 4:N_max
    for n in 1:N
        optimization_step_test(N, n, T)
    end
end

# The whole network is one `GeometricOptimizers.TrainingOptimizer`: one cache and one state over the
# whole parameter set, and not one per layer as before 0.9.
@testset "one cache and one state over the whole network" begin
    model = Chain(StiefelLayer(6, 3), Dense(6, 6, tanh))
    nn = NeuralNetwork(model, KernelAbstractions.CPU(), Float32)
    o = Optimizer(AdamOptimizer(), nn)

    @test o.training isa GeometricOptimizers.TrainingOptimizer
    @test o.training.cache isa GeometricOptimizers.AdamCache{Float32}
    @test o.training.state isa AdamState{Float32}
    @test keys(GeometricOptimizers.solution(o.training.state)) ==
          keys(GeometricMachineLearning.params(nn))
end
