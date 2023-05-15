using MLDatasets 
using Random
using GeometricMachineLearning
using LinearAlgebra
using ProgressMeter
using Lux

#Lux is needed for this flatten operation -> should be removed!
import Flux, Zygote

train_x, train_y = MNIST(split=:train)[:]
test_x, test_y = MNIST(split=:test)[:]

#for visualization
#using ImageInTerminal, ImageShow
#convert2image(MNIST, train_x[:,:,1])

train_x = Flux.flatten(train_x) #|> gpu
train_y = Flux.onehotbatch(train_y, 0:9) #|> gpu
test_x = Flux.flatten(test_x) #|> gpu
test_y = Flux.onehotbatch(test_y, 0:9) #|> gpu

#encoder layer
Ψᵉ = Chain(
    Dense(28*28, 64, tanh),
    Dense(64, 64, tanh),
    Dense(64, 10, Lux.σ)
    )

ps, st = Lux.setup(Random.default_rng(), Ψᵉ)  .|> gpu

#loss_sing
function loss_sing(ps, x, y)
    norm(Lux.apply(Ψᵉ, x, ps, st)[1] - y)
end
function loss_sing(ps, train_x, train_y, index)
    loss_sing(ps, train_x[:, index] |>gpu, train_y[:, index] |> gpu)    
end
function full_loss(ps, train_x, train_y)
    num = size(train_x, 2)
    mapreduce(index -> loss_sing(ps, train_x, train_y, index), +, 1:num)
end

num = size(train_x,2)
batch_size = 64
training_steps = 100


o = AdamOptimizer()
cache = init_optimizer_cache(Ψᵉ, o)

println("initial loss: ", full_loss(ps, train_x, train_y)/num)

@showprogress "Training network ..." for i in 1:training_steps
    index₁ = Int(ceil(rand()*num))
    x = train_x[:, index₁] |> gpu
    y = train_y[:, index₁] |> gpu
    l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
    dp = pb(one(l))[1]
    #dp = Zyogte.gradient(ps -> loss_sing(ps, x, y), ps)[1]

    indices = Int.(ceil.(rand(batch_size -1)*num))
    for index in indices
        x = train_x[:, index] |> gpu
        y = train_y[:, index] |> gpu
        l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
        dp = _add(dp, pb(one(l))[1])
    end
    optimization_step!(o, Ψᵉ, ps, cache, dp)
end
println("final loss: ", full_loss(ps, train_x, train_y)/num)

println("\nfinal test loss: ", full_loss(ps, test_x, test_y)/size(test_x, 2))
