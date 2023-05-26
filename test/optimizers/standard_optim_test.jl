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
optim = StandardOptimizer(1e-3)
g = Zygote.gradient(f, ps)[1]
ps1 = deepcopy(ps)

#This has to be changed to work with the new optimizers syntax!
#=
apply!(optim, nothing, model, ps1, g)
new_val = f(ps1)
@printf "Before optimization: %.5e. " old_val
@printf "After optimization: %.5e" new_val

for layer_number in 1:length(model)
    for key in keys(ps[layer_number])
        ps[layer_number][key] .-= 1e-3 * g[layer_number][key]
    end
end
new_val_manual = f(ps)
@printf "After manuel optimization %.5e" new_val_manual
=#