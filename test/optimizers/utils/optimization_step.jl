using GeometricMachineLearning
using Test
using LinearAlgebra
import Lux
import Random

struct StiefelTestLayer <: Lux.AbstractExplicitLayer
    N::Integer 
    n::Integer
end 

function Lux.initialparameters(rng::Random.AbstractRNG, d::StiefelTestLayer)
    (Y=rand(rng, StiefelManifold{Float32}, d.N, d.n), )
end

function GeometricMachineLearning.retraction(::StiefelTestLayer, B::NamedTuple)
    GeometricMachineLearning.geodesic(B)
end

function optimization_step(N, n)
    model = Lux.Chain(StiefelTestLayer(N, n), Lux.Dense(n, n, tanh))
    ps, _ = Lux.setup(Random.default_rng(), model)
    # gradient 
    dx = (layer_1=(Y=rand(Float32, N, n),), layer_2=(weight=rand(Float32, n, n), bias=rand(Float32, n, 1)))
    m = AdamOptimizer()
    # randomize the cache!
    o = Optimizer(m, ps)

    ps2 = deepcopy(ps)
    optimization_step!(o, model, ps, dx)
    @test typeof(ps.layer_1.Y) <: StiefelManifold
    for key1 in keys(ps)
        for key2 in keys(ps[key1])
            @test norm(ps[key1][key2] - ps2[key1][key2]) > 1f-6
        end
    end
end

N_max = 10
for N = 4:N_max
    for n = 1:N
        optimization_step(N, n)
    end
end