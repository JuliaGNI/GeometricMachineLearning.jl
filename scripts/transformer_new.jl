"""
TODO: Add a better predictor at the end! It should set the biggest value of the softmax to 1 and the rest to zero!
"""

using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots, CUDA
import Zygote, MLDatasets

# MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 5
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# call data loader
dl = DataLoader(train_x, train_y, batch_size=128)
dl_test = DataLoader(test_x, test_y, batch_size=length(test_y))

# Implement classification layer!!!

#encoder layer - final layer has to be added for evaluation purposes!
Ψᵉ₁ = Lux.Chain(
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=false),
    Lux.Dense(patch_length^2, 10, Lux.σ, use_bias=false)
)

Ψᵉ₂ = Lux.Chain(
    #Embedding(patch_length^2, patch_number),
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=true),
    Lux.Dense(patch_length^2, 10, Lux.σ, use_bias=false)
)

function loss_sing(Ψᵉ, ps, st, x, y)
    norm(sum(Lux.apply(Ψᵉ, x, ps, st)[1], dims=2)/size(x, 2) - y)
end

function loss_sing(Ψᵉ, ps, st, train_x, train_y, index)
    loss_sing(Ψᵉ, ps, st, train_x[index], train_y[index])    
end
    
function full_loss(Ψᵉ, ps, st, train_x, train_y)
    num = length(train_x)
    mapreduce(index -> loss_sing(Ψᵉ, ps, st, train_x, train_y, index), +, 1:num)    
end


#err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::Lux.Chain, batch_size=64, training_steps=10000, o=AdamOptimizer())
    ps, st = Lux.setup(CUDA.device(), Random.default_rng(), Ψᵉ) 

    num = length(train_x)

    optimizer_instance = Optimizer(o, ps)

    println("initial test loss: ", full_loss(Ψᵉ, ps, st, test_x, test_y)/length(test_x), "\n")

    @showprogress "Training network ..." for i in 1:training_steps
        index₁ = Int(ceil(rand()*num))
        x = train_x[index₁]
        y = train_y[index₁] 
        l, pb = Zygote.pullback(ps -> loss_sing(Ψᵉ, ps, st, x, y), ps)
        dp = pb(one(l))[1]

        indices = Int.(ceil.(rand(batch_size -1)*num))
        for index in indices
            x = train_x[index] 
            y = train_y[index] 
            l, pb = Zygote.pullback(ps -> loss_sing(Ψᵉ, ps, st, x, y), ps)
            dp = _add(dp, pb(one(l))[1])
        end

        optimization_step!(optimizer_instance, Ψᵉ, ps, dp)    
    end

    println("final test loss: ", full_loss(Ψᵉ, ps, st, test_x, test_y)/length(test_x), "\n")

    ps
end

batch_size = 100
epochs = 10
training_steps = Int(ceil(length(train_x)*epochs/batch_size))

ps₁ = transformer_training(Ψᵉ₁, batch_size, training_steps, AdamOptimizer())
ps₂ = transformer_training(Ψᵉ₂, batch_size, training_steps, AdamOptimizer())

#loss_array₃ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, StandardOptimizer(0.001))
#loss_array₄ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, MomentumOptimizer(0.001, 0.5))
