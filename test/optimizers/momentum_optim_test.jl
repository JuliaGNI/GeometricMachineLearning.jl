using GeometricMachineLearning
using Lux
using Random
using Zygote
using Printf

model = Lux.Chain(Lux.Dense(4, 3), Lux.Dense(3, 1))
ps, st = Lux.setup(Random.default_rng(), model)

random_element = randn(4)
function f(p)
    sum(model(random_element, p, st)[1])^2
end

old_val = f(ps)

optim = MomentumOptimizer(1e-3,5e-1)
g = Zygote.gradient(f, ps)[1]
cache = MomentumOptimizerCache(optim, model, ps, g)

post_init_val = f(ps)

apply!(optim, cache, model, ps, g)
new_val = f(ps)
@printf "Before optimization: %.5e. " old_val
@printf "After initialization: %.5e. " post_init_val
@printf "After optimization: %.5e. " new_val


