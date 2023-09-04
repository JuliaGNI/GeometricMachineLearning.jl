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
batch_size = 128

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# call data loader
dl = DataLoader(train_x, train_y, batch_size=batch_size)
dl_test = DataLoader(test_x, test_y, batch_size=length(test_y))

#encoder layer - final layer has to be added for evaluation purposes!
Ψᵉ₁ = Lux.Chain(
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=false),
    Classification(patch_length^2, 10, σ)
)

Ψᵉ₂ = Lux.Chain(
    #Embedding(patch_length^2, patch_number),
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=true),
    Classification(patch_length^2, 10, σ)
)


# err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::Lux.Chain; backend=CPU(), training_steps=10000, o=AdamOptimizer())
    ps = initialparameters(backend, eltype(dl.data), Ψᵉ) 

    optimizer_instance = Optimizer(o, ps)

    draw_batch!(dl_test)
    println("initial test loss: ", loss(Ψᵉ, ps, dl_test), "\n")

    @showprogress "Training network ..." for i in 1:training_steps
        draw_batch!(dl)
        l, pb = Zygote.pullback(ps -> loss(Ψᵉ, ps, dl), ps)
        dp = pb(one(l))[1]

        optimization_step!(optimizer_instance, Ψᵉ, ps, dp)    
    end

    println("final test loss: ", loss(Ψᵉ, ps, dl_test), "\n")

    ps
end

n_epochs = 10
# calculate number of epochs
training_steps = Int(ceil(length(train_x)*n_epochs/batch_size))

ps₁ = transformer_training(Ψᵉ₁, batch_size, training_steps, AdamOptimizer())
ps₂ = transformer_training(Ψᵉ₂, batch_size, training_steps, AdamOptimizer())

#loss_array₃ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, StandardOptimizer(0.001))
#loss_array₄ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, MomentumOptimizer(0.001, 0.5))
