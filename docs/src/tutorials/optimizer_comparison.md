# Comparison of Optimizers

```@example comparison
using GeometricMachineLearning
using GLMakie
import Random
Random.seed!(123)

f(x::Number, y::Number) = x ^ 2 + y ^ 2
function make_surface()
    n = 100
    r = √2
    u = range(-π, π; length = n)
    v = range(0, π; length = n)
    x = r * cos.(u) * sin.(v)'
    y = r * sin.(u) * sin.(v)'
    z = f.(x, y)
    x, y, z
end

fig = Figure()
ax = Axis3(fig[1, 1])
surface!(ax, make_surface()...; alpha = .3, transparency = true)

init_con = rand(2, 1)
init_cont = Tuple(init_con)
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
scatter!(ax, init_cont..., f(init_cont...); color = mred, marker = :star5)

weights = (xy = init_con, )
η = 1e-3
method1 = GradientOptimizer(η)
method2 = AdamOptimizer(η)
method3 = BFGSOptimizer(η)
optimizer1 = Optimizer(method1, weights)
optimizer2 = Optimizer(method2, weights)
```