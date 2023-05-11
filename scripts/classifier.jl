using MLDatasets 
using Random
using GeometricMachineLearning
using LinearAlgebra

using Lux
#Lux is needed for this flatten operation -> should be removed!
import Flux, Zygote

train_x, train_y = MNIST.traindata()
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

ps, st = Lux.setup(Random.default_rng(), Ψᵉ)  .|> gpu

#loss_sing
function loss_sing(ps, x, y)
    norm(Lux.apply(Ψᵉ, x, ps, st)[1] - y)
end
function loss_sing(ps, train_x, train_y, index)
    loss_sing(ps, train_x[:, index], train_y[:, index])    
end
function loss(ps, train_x, train_y, batch_size=10)
    num = size(train_x,2)
    indices = Int.(ceil.(rand(batch_size)*num))
    mapreduce(index -> loss_sing(ps, train_x, train_y, index), +, indices)
end

function full_loss(ps, train_x, train_y)
    num = size(train_x, 2)
    mapreduce(index -> loss_sing(ps, train_x, train_y, index), +, 1:num)
end

o = AdamOptimizer()
cache = init_optimizer_cache(Ψᵉ, o)
println("initial loss: ", full_loss(ps, train_x, train_y))

training_steps = 1000000

loss_closure(ps) = loss(ps, train_x, train_y)
num = size(train_x,2)
batch_size = 5

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
        dp += pb(one(l))[1]
    end
    optimization_step!(o, Ψᵉ, ps, cache, dp)
end
println("final loss: ", full_loss(ps, train_x, train_y))
