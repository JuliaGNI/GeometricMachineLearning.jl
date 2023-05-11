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

train_x = Flux.flatten(train_x) # .|> gpu
train_y = Flux.onehotbatch(train_y, 0:9)

#encoder layer
Ψᵉ = Chain(
    Dense(28*28, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16,10, Lux.σ)
    )

ps, st = Lux.setup(Random.default_rng(), Ψᵉ) # .|> gpu

#loss_sing
function loss_sing(ps, train_x, train_y, index)
    norm(Lux.apply(Ψᵉ, train_x[:, index], ps, st)[1] - train_y[:, index])
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

training_steps = 3
loss_closure(ps) = loss(ps, train_x, train_y)
for i in 1:training_steps
    #@time dp = Zygote.gradient(loss_closure, ps)[1]
    @time l, pb = Zygote.pullback(loss_closure, ps)
    @time dp = pb(one(l))[1]
    optimization_step!(o, Ψᵉ, ps, cache, dp)
end
println("final loss: ", full_loss(ps, train_x, train_y))
