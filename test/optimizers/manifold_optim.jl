using GeometricMachineLearning
using Lux
using Random
using Zygote
using LinearAlgebra

m = 20
n = 200
x = rand(2 * n)

model = Chain(Gradient(2 * n, 4 * n), Gradient(2 * n), SymplecticStiefelLayer(2 * m, 2 * n; inverse = true))

n_runs = Int(5e3)
err_vec = zeros(n_runs+1)
err_vec2 = zeros(n_runs+1)

### Test for StandardOptimizer
#note! optimizer and network state are not the same!
optim = StandardOptimizer(1e-5)
ps, st = Lux.setup(Random.default_rng(), model)
err_vec[1] = norm(Lux.apply(model,x,ps,st)[1])
@time for i in 1:n_runs
    g = gradient(p -> norm(Lux.apply(model, x, p, st)[1]), ps)[1]
    apply!(optim, nothing, model, ps, g)
    err_vec[i+1] = norm(Lux.apply(model,x,ps,st)[1])
end
print(norm(Lux.apply(model,x,ps,st)[1]))

### Test for MomentumOptimizer
optim = MomentumOptimizer(1e-5,1e-2)
ps, st = Lux.setup(Random.default_rng(), model)
#hacky for the moment!!!!!!!!!1 fix!!!!!
model2 = Chain(Gradient(2 * n, 4 * n), Gradient(2 * n), SymplecticStiefelLayer(2 * n, 2 * n; inverse = true))
state = init_momentum(model2)
err_vec2[1] = norm(Lux.apply(model,x,ps,st)[1])
@time for i in 1:n_runs
    g = gradient(p -> norm(Lux.apply(model, x, p, st)[1]), ps)[1]
    apply!(optim, state, model, ps, g)
    err_vec2[i+1] = norm(Lux.apply(model,x,ps,st)[1])
end
print(norm(Lux.apply(model,x,ps,st)[1]))
