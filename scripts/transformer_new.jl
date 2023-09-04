"""
TODO: Add a better predictor at the end! It should set the biggest value of the softmax to 1 and the rest to zero!
"""

using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots, CUDA
using AbstractNeuralNetworks
import Zygote, MLDatasets

# remove this after AbstractNeuralNetworks PR has been merged 
GeometricMachineLearning.Chain(model::Chain, d::AbstractNeuralNetworks.AbstractExplicitLayer) = Chain(model.layers..., d)

# MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 7
patch_number = (image_dim÷patch_length)^2
batch_size = 128
activation = σ
n_epochs = 1
backend = CPU()

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

#encoder layer - final layer has to be added for evaluation purposes!
Ψᵉ₁ = Chain(
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=false),
    Classification(patch_length^2, 10, activation)
)

Ψᵉ₂ = Chain(
    #Embedding(patch_length^2, patch_number),
    Transformer(patch_length^2, n_heads, n_layers, Stiefel=true),
    Classification(patch_length^2, 10, activation)
)


# err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::Chain; backend=CPU(), n_training_steps=10000, o=AdamOptimizer())
    # call data loader
    dl = DataLoader(train_x, train_y, batch_size=batch_size)
    dl_test = DataLoader(test_x, test_y, batch_size=length(test_y))

    ps = initialparameters(backend, eltype(dl.data), Ψᵉ) 

    optimizer_instance = Optimizer(o, ps)

    println("initial test loss: ", loss(Ψᵉ, ps, dl_test), "\n")

    progress_object = Progress(n_training_steps; enabled=true)

    for i in 1:n_training_steps
        redraw_batch!(dl)
        loss_val, pb = Zygote.pullback(ps -> loss(Ψᵉ, ps, dl), ps)
        dp = pb(one(loss_val))[1]

        optimization_step!(optimizer_instance, Ψᵉ, ps, dp)
        ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_val)])   
    end

    println("final test loss: ", loss(Ψᵉ, ps, dl_test), "\n")

    ps
end

# calculate number of epochs
n_training_steps = Int(ceil(length(train_y)*n_epochs/batch_size))

ps₁ = transformer_training(Ψᵉ₁, backend=backend, n_training_steps=n_training_steps)
ps₂ = transformer_training(Ψᵉ₂, backend=backend, n_training_steps=n_training_steps)

#loss_array₃ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, StandardOptimizer(0.001))
#loss_array₄ = transformer_training(Ψᵉ₂, batch_size, training_steps, err_freq, MomentumOptimizer(0.001, 0.5))
