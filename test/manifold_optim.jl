using GeometricMachineLearning
using Lux
using Random
using Zygote

m = 20
n = 200
x = rand(n)

optim = StandardOptimizer(1e-5)
state = init(optim, x)
model = Chain(Gradient(n, 2n), SymplecticStiefelLayer(m, n; inverse = true))

ps, st = Lux.setup(Random.default_rng(), model)
g = gradient(p -> sum(Lux.apply(model, x, p, st)[1]), ps)[1]
@time apply!(optim, state, model, ps, g)

ps, st = Lux.setup(Random.default_rng(), model)
g = gradient(p -> sum(Lux.apply(model, x, p, st)[1]), ps)[1]
@time apply!(optim, state, model, ps, g)


# Tests

optim = StandardOptimizer()

layer = SymplecticStiefelLayer(m, n; inverse = true)
ps, st = Lux.setup(Random.default_rng(), layer)
x = rand(n)
g = gradient(p -> sum(Lux.apply(layer, x, p, st)[1]), ps)[1]


model = Chain(Gradient(200, 1000), Gradient(200, 500),
               SymplecticStiefelLayer(20, 200; inverse = true), Gradient(20, 50))
ps, st = Lux.setup(Random.default_rng(), model)
x₂ = rand(200)
g₂ = gradient(p -> sum(Lux.apply(model, x₂, p, st)[1]), ps)[1]

state = init(optim, x₂)
apply!(optim, state, model, ps, g₂)
