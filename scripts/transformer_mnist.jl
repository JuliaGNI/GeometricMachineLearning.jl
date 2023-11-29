"""
TODO: Add a better predictor at the end! It should set the biggest value of the softmax to 1 and the rest to zero!
"""

using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots, CUDA
import Zygote, MLDatasets

# MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
transformer_dim = 49
n_heads = 7
n_layers = 16
number_of_patch = (image_dim÷patch_length)^2
batch_size = 2048
activation = softmax
n_epochs = 500
add_connection = false

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# use CUDA backend if available. else use CPU()
backend, train_x, test_x, train_y, test_y = 
    try
        error()
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

dl = DataLoader(train_x, train_y)
dl_test = DataLoader(test_x, test_y)

# the difference between the first and the second model is that we put the weights on the Stiefel manifold in the second case
model1 = ClassificationTransformer(dl, n_heads=n_heads, n_layers=n_layers, Stiefel=false, add_connection=add_connection)
model2 = ClassificationTransformer(dl, n_heads=n_heads, n_layers=n_layers, Stiefel=true, add_connection=add_connection)

batch = Batch(batch_size)

# err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::GeometricMachineLearning.Architecture; n_epochs=100, opt=AdamOptimizer())
    nn = NeuralNetwork(Ψᵉ, backend, eltype(dl))
    optimizer_instance = Optimizer(opt, nn)

    println("initial test accuracy: ", GeometricMachineLearning.accuracy(nn, dl_test), "\n")

    progress_object = Progress(n_epochs; enabled=true)

    # use the `time` function to get the system time.
    init_time = time()
    total_time = init_time - time()

    loss_array = zeros(eltype(train_x), n_epochs)
    for i in 1:n_epochs
        # there is some functionality in a recent PR that streamlines some of this -> make sure to include this!
        loss_val = optimize_for_one_epoch!(optimizer_instance, nn, dl, batch)

        ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_val)])   
        loss_array[i] = loss_val

        # update runtime
        total_time = init_time - time()
    end

    accuracy_score = GeometricMachineLearning.accuracy(nn, dl_test)
    println("final test accuracy: ", accuracy_score, "\n")

    loss_array, ps, total_time, accuracy_score
end


loss_array2, ps2, total_time2, accuracy_score2 = transformer_training(model2, n_epochs=n_epochs)
loss_array1, ps1, total_time1, accuracy_score1 = transformer_training(model1, n_epochs=n_epochs)
loss_array3, ps3, total_time3, accuracy_score3 = transformer_training(model2, n_epochs=n_epochs, opt=GradientOptimizer(0.001))
loss_array4, ps4, total_time4, accuracy_score4 = transformer_training(model2, n_epochs=n_epochs, opt=MomentumOptimizer(0.001, 0.5))

p1 = plot(loss_array1, color=1, label="Regular weights", ylimits=(0.,1.4), linewidth=2)
plot!(p1, loss_array2, color=2, label="Weights on Stiefel Manifold", linewidth=2)
png(p1, "Stiefel_Regular")

p2 = plot(loss_array2, color=2, label="Adam", ylimits=(0.,1.4), linewidth=2)
plot!(p2, loss_array3, color=1, label="Gradient", linewidth=2)
plot!(p2, loss_array4, color=3, label="Momentum", linewidth=2)
png(p2, "Adam_Gradient_Momentum")

text_string = 
    "n_epochs: " * string(n_epochs) * "\n"
    "Regular weights:   time: " * string(total_time1) * " classification accuracy: " * string(accuracy_score1) * "\n" *
    "Stiefel weights:   time: " * string(total_time2) * " classification accuracy: " * string(accuracy_score2) * "\n" *
    "GradientOptimizer: time: " * string(total_time3) * " classification accuracy: " * string(accuracy_score3) * "\n" *
    "MomentumOptimizer: time: " * string(total_time4) * " classification accuracy: " * string(accuracy_score4) * "\n"

display(text_string)

open("measure_times"*string(backend), "w") do file
    write(file, text_string)
end
