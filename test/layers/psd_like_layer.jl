using GeometricMachineLearning, Test
import Random, Lux

N = 10
n = 5

model = PSDLayer(2*N, 2*n)

ps, st = Lux.setup(Random.default_rng(), model)
@test typeof(ps.weight) <: StiefelManifold

ps, st = Lux.setup(TrivialInitRNG(), model)
@test typeof(ps.weight) <: StiefelLieAlgHorMatrix