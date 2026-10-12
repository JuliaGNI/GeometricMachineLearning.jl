using GeometricMachineLearning, Test, LinearAlgebra, KernelAbstractions
import GeometricOptimizers
import GeometricOptimizers: optimization_step!
import GeometricMachineLearning: NeuralNetwork
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

function optimization_step_test(N, n, T)
    model = Chain(StiefelLayer(N, n), Dense(N, N, tanh))
    ps = NeuralNetwork(model, KernelAbstractions.CPU(), T).params
    # gradient
    dx = NetworkParameters((L1 = (weight = rand(T, N, n),),
        L2 = (W = rand(T, N, N), b = rand(T, N))))
    o = NetworkOptimizer(ps; algorithm = Adam())

    ps2 = deepcopy(ps)
    optimization_step!(ps, o, dx)
    @test ps[1].weight isa StiefelManifold{T}
    for (layers1, layers2) in zip(values(ps), values(ps2))
        for key in keys(layers1)
            @test eltype(layers1[key]) === T
            @test norm(layers1[key] - layers2[key]) > T(1.0f-6)
        end
    end
end

@testset "one Adam step moves every leaf of the network ($T)" for T in REAL_ELTYPES
    for N in 4:10
        for n in 1:N
            optimization_step_test(N, n, T)
        end
    end
end

# One cache and one state cover the whole network: their solution is a parameter set with the
# network's layers as its keys, not one cache per layer.
@testset "one cache and one state over the whole network ($T)" for T in REAL_ELTYPES
    model = Chain(StiefelLayer(6, 3), Dense(6, 6, tanh))
    nn = NeuralNetwork(model, KernelAbstractions.CPU(), T)
    o = NetworkOptimizer(nn; algorithm = Adam())

    @test o.optimizer.cache isa GeometricOptimizers.OptimizerCache{T}
    @test o.optimizer.state isa GeometricOptimizers.OptimizerState{T}
    for x in (GeometricOptimizers.solution(o.optimizer.cache),
        GeometricOptimizers.solution(o.optimizer.state))
        @test x isa NetworkParameters
        @test keys(x) == keys(GeometricMachineLearning.params(nn))
        @test x.L1.weight isa StiefelManifold{T}
        @test eltype(x.L2.W) === eltype(x.L2.b) === T
    end
end

# A function barrier with concrete argument types, which runs the call once before it measures.
function allocations(f::F, a::A, b::B, c::C) where {F, A, B, C}
    f(a, b, c)
    @allocated f(a, b, c)
end

# The training step is the hot path of `train!`. It is inferred, and on a Euclidean layer it allocates
# nothing. With a Stiefel leaf `GeometricOptimizers`' own step allocates (about 2 to 4 kB here), and
# the step through `NetworkOptimizer` allocates the same bytes, within 32.
@testset "the step through NetworkOptimizer is inferred and allocates nothing of its own ($T)" for T in REAL_ELTYPES
    for algorithm in (GradientMethod(), MomentumMethod(), Adam())
        ps = NetworkParameters((L1 = (W = rand(T, 3, 3), b = rand(T, 3)),))
        dp = NetworkParameters((L1 = (W = rand(T, 3, 3), b = rand(T, 3)),))
        o = NetworkOptimizer(ps; algorithm)
        @inferred optimization_step!(ps, o, dp)
        @test allocations(optimization_step!, ps, o, dp) == 0
        @test eltype(ps.L1.W) === eltype(ps.L1.b) === T

        ps = NetworkParameters((L1 = (W = rand(T, 3, 3), b = rand(T, 3)),
            L2 = (Y = rand(StiefelManifold{T}, 4, 2),)))
        dp = NetworkParameters((L1 = (W = rand(T, 3, 3), b = rand(T, 3)),
            L2 = (Y = rand(T, 4, 2),)))
        o = NetworkOptimizer(ps; algorithm)
        @inferred optimization_step!(ps, o, dp)
        @test abs(allocations(optimization_step!, ps, o, dp) -
                  allocations(optimization_step!, ps, o.optimizer, dp)) ≤ 32
        @test ps.L2.Y isa StiefelManifold{T}
    end
    @test allocations((a, b, c) -> collect(a), 1:3, 1, 1) > 0
end
