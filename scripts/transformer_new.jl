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
n_layers = 10
number_of_patch = (image_dim÷patch_length)^2
batch_size = 2048
activation = softmax
n_epochs = 100
add_connection = false

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# use CUDA backend if available. else use CPU()
backend, train_x, test_x, train_y, test_y = 
    try
        CUDABackend(),
        train_x |> cu,
        test_x |> cu,
        train_y |> cu,
        test_y |> cu
    catch
        CPU(), 
        train_x, 
        test_x, 
        train_y, 
        test_y
end


#encoder layer - final layer has to be added for evaluation purposes!
model1 = Chain(Transformer(patch_length^2, n_heads, n_layers, Stiefel=false, add_connection=add_connection),
	    Classification(patch_length^2, 10, activation))

model2 = Chain(Transformer(patch_length^2, n_heads, n_layers, Stiefel=true, add_connection=add_connection),
	    Classification(patch_length^2, 10, activation))

# err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::Chain; backend=CPU(), n_epochs=100, opt=AdamOptimizer())
    # call data loader
    dl = DataLoader(train_x, train_y)
    dl_test = DataLoader(test_x, test_y)
    batch = Batch(batch_size)

    ps = initialparameters(backend, eltype(dl.input), Ψᵉ) 

    optimizer_instance = Optimizer(opt, ps)

    println("initial test accuracy: ", GeometricMachineLearning.accuracy(Ψᵉ, ps, dl_test), "\n")

    progress_object = Progress(n_epochs; enabled=true)

    # use the `time` function to get the system time.
    init_time = time()
    total_time = init_time - time()

    loss_array = zeros(eltype(train_x), n_epochs)
    for i in 1:n_epochs
        loss_val = optimize_for_one_epoch!(optimizer_instance, Ψᵉ, ps, dl, batch)

        ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_val)])   
        loss_array[i] = loss_val

        # update runtime
        total_time = init_time - time()
    end

    println("final test accuracy: ", GeometricMachineLearning.accuracy(Ψᵉ, ps, dl_test), "\n")

    loss_array, ps, total_time
end


loss_array2, ps2, total_time2 = transformer_training(model2, backend=backend, n_epochs=n_epochs)
loss_array1, ps1, total_time1 = transformer_training(model1, backend=backend, n_epochs=n_epochs)
loss_array3, ps3, total_time3 = transformer_training(model2, backend=backend, n_epochs=n_epochs, opt=GradientOptimizer(0.001))
loss_array4, ps4, total_time4 = transformer_training(model2, backend=backend, n_epochs=n_epochs, opt=MomentumOptimizer(0.001, 0.5))

p1 = plot(loss_array1, color=1, label="Regular weights", ylimits=(0.,1.4))
plot!(p1, loss_array2, color=2, label="Weights on Stiefel Manifold")
png(p1, "Stiefel_Regular")

p2 = plot(loss_array2, color=2, label="Adam", ylimits=(0.,1.4))
plot!(p2, loss_array3, color=1, label="Gradient")
plot!(p2, loss_array4, color=3, label="Momentum")
png(p2, "Adam_Gradient_Momentum")

display(
    "Regular weights:   ", total_time1, "\n",
    "Stiefel weights:   ", total_time2, "\n",
    "GradientOptimizer: ", total_time3, "\n",
    "MomentumOptimizer: ", total_time4, "\n"
    )