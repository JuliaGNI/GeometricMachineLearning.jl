using GeometricMachineLearning, Test, LinearAlgebra, KernelAbstractions
using AbstractNeuralNetworks: AbstractExplicitLayer
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