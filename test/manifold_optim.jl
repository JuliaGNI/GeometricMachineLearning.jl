using Lux
using Zygote
using Random

include("../src/optimizers/AbstractOptimizer.jl")
include("../src/optimizers/StandardOptimizer.jl")
include("../src/layers/gradient.jl")
include("../src/layers/manifold_layer.jl")


model = Chain(Gradient(10,20), SymplecticStiefelLayer(4,10;inverse=true))
ps, st = Lux.setup(Random.default_rng(), model)

o = StandardOptimizer(1e-5)

g = gradient(p -> sum(Lux.apply(model, rand(10), p, st)[1]), ps)[1]

apply!(o, ps, g, st)
