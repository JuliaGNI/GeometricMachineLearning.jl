using GeometricMachineLearning, LinearAlgebra, ProgressMeter
import Lux, Zygote, Random, MLDatasets, Flux

#MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 5
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]


#preprocessing steps 
train_x =   Tuple(map(i -> split_and_flatten(train_x[:,:,i], patch_length), 1:size(train_x,3)))
test_x =    Tuple(map(i -> split_and_flatten(test_x[:,:,i], patch_length), 1:size(test_x,3)))

#implement this encoding yourself!
train_y = Flux.onehotbatch(train_y, 0:9) #|> gpu
test_y = Flux.onehotbatch(test_y, 0:9) #|> gpu


#encoder layer - final layer has to be added for evaluation purposes!
Ψᵉ = Lux.Chain(
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=true),
    Lux.Dense(patch_length^2, 10, Lux.σ, use_bias=false)
)

ps, st = Lux.setup(Random.default_rng(), Ψᵉ) # .|> gpu  

#loss_sing
function loss_sing(ps, x, y)
    norm(Lux.apply(Ψᵉ, x, ps, st)[1][:,1] - y)
end
function loss_sing(ps, train_x, train_y, index)
    loss_sing(ps, train_x[index], train_y[:, index])    
end
function full_loss(ps, train_x, train_y)
    num = length(train_x)
    mapreduce(index -> loss_sing(ps, train_x, train_y, index), +, 1:num)
end

num = length(train_x)
batch_size = 64
training_steps = 10

o = AdamOptimizer()
cache = init_optimizer_cache(Ψᵉ, o)

println("initial loss: ", full_loss(ps, train_x, train_y)/num)

@showprogress "Training network ..." for i in 1:training_steps
    index₁ = Int(ceil(rand()*num))
    x = train_x[index₁] #|> gpu
    y = train_y[:, index₁] #|> gpu
    l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
    dp = pb(one(l))[1]
    #dp = Zyogte.gradient(ps -> loss_sing(ps, x, y), ps)[1]

    indices = Int.(ceil.(rand(batch_size -1)*num))
    for index in indices
        x = train_x[index] #|> gpu
        y = train_y[:, index] #|> gpu
        l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
        dp = _add(dp, pb(one(l))[1])
    end
    optimization_step!(o, Ψᵉ, ps, cache, dp)
end
println("final loss: ", full_loss(ps, train_x, train_y)/num)

println("\nfinal test loss: ", full_loss(ps, test_x, test_y)/length(test_x))