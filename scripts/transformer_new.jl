"""
TODO: Add a better predictor at the end! It should set the biggest value of the softmax to 1 and the rest to zero!
"""

using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots, CUDA
using AbstractNeuralNetworks
import Zygote, MLDatasets

# remove this after AbstractNeuralNetworks PR has been merged 
GeometricMachineLearning.Chain(model::Chain, d::AbstractNeuralNetworks.AbstractExplicitLayer) = Chain(model.layers..., d)
GeometricMachineLearning.Chain(d::AbstractNeuralNetworks.AbstractExplicitLayer, model::Chain) = Chain(d, model.layers...)

# MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
transformer_dim = 49
n_heads = 7
n_layers = 16
number_of_patch = (image_dim÷patch_length)^2
batch_size = 512
activation = softmax
n_epochs = 2
add_connection = false
backend = CUDABackend()

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]
if backend == CUDABackend()
	train_x = train_x |> cu 
	test_x = test_x |> cu 
	train_y = train_y |> cu 
	test_y = test_y |> cu
end

#encoder layer - final layer has to be added for evaluation purposes!
model1 = Chain(Transformer(patch_length^2, n_heads, n_layers, Stiefel=false, add_connection=add_connection),
	    Classification(patch_length^2, 10, activation))

model2 = Chain(Transformer(patch_length^2, n_heads, n_layers, Stiefel=true, add_connection=add_connection),
	    Classification(patch_length^2, 10, activation))


# err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::Chain; backend=CPU(), n_training_steps=10000, o=AdamOptimizer())
    # call data loader
    dl = DataLoader(train_x, train_y, batch_size=batch_size)
    dl_test = DataLoader(test_x, test_y, batch_size=length(test_y))

    ps = initialparameters(backend, eltype(train_x), Ψᵉ) 

    optimizer_instance = Optimizer(o, ps)

    println("initial test loss: ", loss(Ψᵉ, ps, dl_test), "\n")

    progress_object = Progress(n_training_steps; enabled=true)

    loss_array = KernelAbstractions.zeros(backend, eltype(train_x), n_training_steps)
    for i in 1:n_training_steps
        redraw_batch!(dl)
        loss_val, pb = Zygote.pullback(ps -> loss(Ψᵉ, ps, dl), ps)
        dp = pb(one(loss_val))[1]

        optimization_step!(optimizer_instance, Ψᵉ, ps, dp)
        ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_val)])   
        loss_array[i] = loss_val
    end

    println("final test loss: ", loss(Ψᵉ, ps, dl_test), "\n")

    ps, loss_array
end

# calculate number of epochs
n_training_steps = Int(ceil(length(train_y)*n_epochs/batch_size))

ps₁, loss_array₁ = transformer_training(model1, backend=backend, n_training_steps=n_training_steps)
ps₂, loss_array₂ = transformer_training(model2, backend=backend, n_training_steps=n_training_steps)

ps₃, loss_array₃ = transformer_training(model2, batch_size, training_steps, err_freq, GradientOptimizer(0.001))
ps₄, loss_array₄ = transformer_training(model2, batch_size, training_steps, err_freq, MomentumOptimizer(0.001, 0.5))

p₁ = plot(loss_array₁, color=1, label="Standard")
plot!(p₁, loss_array₂, color=3, label="Stiefel")

p₂ = plot(loss_array₂, color=3, label="Adam")
plot!(p₂, loss_array₃, color=1, label="Gradient")
plot!(p₂, loss_array₄, color=2, label="Momentum")

png(p₁, "Stiefel_Regular")
png(p₂, "Adam_Gradient_Momentum")