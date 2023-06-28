"""
TODO: Add a better predictor at the end! It should set the biggest value of the softmax to 1 and the rest to zero!
"""

using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots
import Lux, Zygote, Random, MLDatasets, Flux, Lux.gpu

#MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 5
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]


#preprocessing steps 
train_x =   Tuple(map(i -> sc_embed(split_and_flatten(train_x[:,:,i], patch_length)) #=|> gpu=#, 1:size(train_x,3)))
test_x =    Tuple(map(i -> sc_embed(split_and_flatten(test_x[:,:,i], patch_length)) #=|> gpu=#, 1:size(test_x,3)))

#implement this encoding yourself!
train_y = Flux.onehotbatch(train_y, 0:9) #|> gpu
test_y = Flux.onehotbatch(test_y, 0:9) #|> gpu


#encoder layer - final layer has to be added for evaluation purposes!
Ψᵉ₁ = Lux.Chain(
    #Embedding(patch_length^2, patch_number),
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=false),
    Lux.Dense(patch_length^2, 10, Lux.σ, use_bias=false)
)

Ψᵉ₂ = Lux.Chain(
    #Embedding(patch_length^2, patch_number),
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=true),
    Lux.Dense(patch_length^2, 10, Lux.σ, use_bias=false)
)

#err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::Lux.Chain, batch_size=64, training_steps=10000, err_freq=100, o=AdamOptimizer())
    ps, st = Lux.setup(Random.default_rng(), Ψᵉ) #.|> gpu  

    #loss_sing
    #note that the loss here is the first column of the output of the entire model; similar to what was done in the ViT paper.
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

    cache = init_optimizer_cache(Ψᵉ, o) 

    loss_array = zeros(training_steps÷err_freq + 1)
    loss_array[1] = full_loss(ps, train_x, train_y)/num
    println("initial loss: ", loss_array[1])

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
        if i%err_freq == 0
            loss_array[1+i÷err_freq] = full_loss(ps, train_x, train_y)/num
        end
    end
    println("final loss: ", loss_array[end])
    println("final test loss: ", full_loss(ps, test_x, test_y)/length(test_x),"\n")

    loss_array
end

batch_size = 16
training_steps = 10000
err_freq = 100
o = AdamOptimizer()

loss_array₁ = transformer_training(Ψᵉ₁, batch_size, training_steps, err_freq, o)
loss_array₂ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, o)

steps = vcat(1:err_freq:training_steps, training_steps+1) .- 1

p₁ = plot(steps, loss_array₁, label="Regular weights", linewidth=2, size=(800,500))
plot!(p₁, steps, loss_array₂, label="Weights on the Stiefel manifold", linewidth=2)
ylims!(0.,2.2)
png(p₁, "transformer_stiefel_reg_comp")


loss_array₃ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, StandardOptimizer(0.001))
loss_array₄ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, MomentumOptimizer(0.001, 0.5))

p₂ = plot(steps, loss_array₃, label="Standard Optimizer",linewidth=2, size=(800,500))
plot!(p₂, steps, loss_array₂, label="Adam Optimizer",linewidth=2)
plot!(p₂, steps, loss_array₄, label="Momentum Optimizer",linewidth=2)
ylims!(0.,2.2)
png(p₂, "transformer_stiefel_ad_mom_stan")