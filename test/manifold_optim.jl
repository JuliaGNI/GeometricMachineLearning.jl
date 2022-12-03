using GeometricMachineLearning
using Lux
using Random
using Zygote

m = 20
n = 200
x = rand(2 * n)

model = Chain(Gradient(2 * n, 4 * n), Gradient(2 * n), SymplecticStiefelLayer(2 * m, 2 * n; inverse = true))

### Test for StandardOptimizer
#note! optimizer and network state are not the same!
optim = StandardOptimizer(1e-3)
ps, st = Lux.setup(Random.default_rng(), model)
g = gradient(p -> sum(Lux.apply(model, x, p, st)[1]), ps)[1]
@time apply!(optim, nothing, model, ps, g)

### Test for MomentumOptimizer
optim = MomentumOptimizer(1e-3,1e-2)
ps, st = Lux.setup(Random.default_rng(), model)
#hacky for the moment!!!!!!!!!1 fix!!!!!
model2 = Chain(Gradient(2 * n, 4 * n), Gradient(2 * n), SymplecticStiefelLayer(2 * n, 2 * n; inverse = true))
state = init_momentum(model2)
g = gradient(p -> sum(Lux.apply(model, x, p, st)[1]), ps)[1]
@time apply!(optim, state, model, ps, g)