using MLDatasets 
using Random
using GeometricMachineLearning
using LinearAlgebra

using Lux
#Lux is needed for this flatten operation -> should be removed!
import Flux, Zygote

train_x, train_y = MNIST(split=:train)[:]
#for visualization
#using ImageInTerminal, ImageShow
#convert2image(MNIST, train_x[:,:,1])

train_x = Flux.flatten(train_x) |> gpu
train_y = Flux.onehotbatch(train_y, 0:9) |> gpu

#encoder layer
Ψᵉ = Chain(
    Dense(28*28, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16,10, Lux.σ)
    )

#decoder layer
Ψᵈ = Chain(
    Dense(10, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16, 28*28, Lux.σ)
)

const model = Chain(Ψᵉ, Ψᵈ)

ps, st = Lux.setup(Random.default_rng(), model) .|> gpu

#loss_sing
function loss_sing(ps, x, y)
    norm(Lux.apply(model, x, ps, st)[1] - x)
end

function loss_sing(ps, train_x, train_y, index)
    loss_sing(ps, train_x[:, index], train_y[:, index])    
end
function full_loss(ps, train_x, train_y)
    num = size(train_x, 2)
    mapreduce(index -> loss_sing(ps, train_x, train_y, index), +, 1:num)
end

o = AdamOptimizer()
cache = init_optimizer_cache(model, o)
println("initial loss: ", full_loss(ps, train_x, train_y))

training_steps = 100

num = size(train_x, 2)
batch_size = 10

for i in 1:training_steps
    #@time dp = Zygote.gradient(loss_closure, ps)[1]

    index₁ = Int(ceil(rand()*num))
    x = train_x[:, index₁]
    y = train_y[:, index₁] 
    l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
    dp = pb(one(l))[1]

    indices = Int.(ceil.(rand(batch_size -1)*num))
    for index in indices
        x = train_x[:, index] 
        y = train_y[:, index] 
        l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
        dp = _add(dp, pb(one(l))[1])
    end
    optimization_step!(o, Ψᵉ, ps, cache, dp)
end
println("final loss: ", full_loss(ps, train_x, train_y))
