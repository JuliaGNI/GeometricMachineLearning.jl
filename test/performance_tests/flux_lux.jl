"""
This compares the performance between Lux and Flux
"""

import Flux, Lux, Zygote, GeometricMachineLearning, Random, LinearAlgebra


data = [([x], 2x-x^3) for x in -2:0.1f0:2]

model = Flux.Chain(Flux.Dense(1 => 23, tanh), Flux.Dense(23 => 1, bias=false), only)

optim = Flux.setup(Flux.Adam(), model)
print("Optimization steps in Flux:\n")
@time for epoch in 1:1000
  Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, optim)
end

model = Lux.Chain(Lux.Dense(1, 23, tanh), Lux.Dense(23, 1, bias=false))
ps, st = Lux.setup(Random.default_rng(), model)

optim = GeometricMachineLearning.AdamOptimizer()
cache = GeometricMachineLearning.init_optimizer_cache(model, optim)

print("\n Optimization steps in Lux: \n")
@time for epoch in 1:1000
    dat = data[1]
    dp = Zygote.gradient(ps -> LinearAlgebra.norm(Lux.apply(model, dat[1], ps, st)[1] .- dat[2])^2, ps)[1]
    data_red = data[2:end]
    for dat in data_red
        dp = GeometricMachineLearning._add(
            dp, 
            Zygote.gradient(ps -> LinearAlgebra.norm(Lux.apply(model, dat[1], ps, st)[1] .- dat[2])^2, ps)[1]
            )
    end
    GeometricMachineLearning.optimization_step!(optim, model, ps, cache, dp)
end