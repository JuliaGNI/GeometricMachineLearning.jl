using GeometricMachineLearning
using GeometricMachineLearning: ResNet
using LinearAlgebra: norm
using ProgressMeter: @showprogress
using Zygote: gradient 
using Plots: plot, plot!
import MLDatasets
import Lux
import Random

image_dim = 28
patch_length = 7
n_heads = 7
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# preprocessing steps (also perform rescaling so that the images have values between 0 and 1)
function preprocess_x(x)
    x_reshaped = zeros(Float32, patch_length^2, patch_number, size(x, 3))
    for i in axes(x, 3)
        x_reshaped[:, :, i] = split_and_flatten(x[:, :, i], patch_length)/255
    end
    x_reshaped
end

train_x_reshaped = preprocess_x(train_x)
test_x_reshaped = preprocess_x(test_x)

# preprocessing/encoding for y
function encode_y(y)
    y_encoded = zeros(Bool, 10, length(y))
    for i in axes(y,1)
        y_encoded[y[i]+1,i] = 1
    end
    y_encoded
end

train_y_encoded = encode_y(train_y)
test_y_encoded = encode_y(test_y)

L = 4

use_softmax=true

#named tuple of models that are compared
models = (
    model₀ = Lux.Chain(Tuple(map(_ -> ResNet(49, tanh), 1:L))..., Classification(patch_length^2, 10, use_bias=false, use_average=false, use_softmax=use_softmax)),
    #model₁ = Lux.Chain( Transformer(patch_length^2, n_heads, L, add_connection=false, Stiefel=false),
    #                    Classification(patch_length^2, 10, use_bias=false, use_average=false, use_softmax=use_softmax)),
    model₂ = Lux.Chain(Transformer(patch_length^2, n_heads, L, add_connection=true, Stiefel=false),
                        Classification(patch_length^2, 10, use_bias=false, use_average=false, use_softmax=use_softmax)),
    #model₃ = Lux.Chain(Transformer(patch_length^2, n_heads, L, add_connection=false, Stiefel=true),
    #                    Classification(patch_length^2, 10, use_bias=false, use_average=false, use_softmax=use_softmax)),
    model₄ = Lux.Chain(Transformer(patch_length^2, n_heads, L, add_connection=true, Stiefel=true),
                     Classification(patch_length^2, 10, use_bias=false, use_average=false, use_softmax=use_softmax))
)

const num = 60000
function training(model::Lux.Chain, batch_size=32, n_epochs=.01, o=AdamOptimizer(), enable_cuda=false, give_training_error=false)
    ps, st = enable_cuda ? Lux.setup(CUDA.device(), Random.default_rng(), model) : Lux.setup(Random.default_rng(), model)
                    
    function loss(ps, x, y)
        x_eval = enable_cuda ? Lux.apply(model, x |> cu, ps, st)[1] : Lux.apply(model, x, ps, st)[1]
        enable_cuda ? norm(x_eval - (y |> cu))/sqrt(size(y, 2)) : norm(x_eval - (y))/sqrt(size(y, 2))
    end
                    
    # the number of training steps is calculated based on the number of epochs and the batch size
    training_steps = Int(ceil(n_epochs*num/batch_size))
    # this records the training error
    loss_array = zeros(training_steps + 1)
    loss_array[1] = give_training_error ? (enable_cuda ? loss(ps, train_x_reshaped |> cu, train_y_encoded |> cu) : loss(ps, train_x_reshaped, train_y_encoded)) : 0.
                    
    give_training_error ? println("initial loss: ", loss_array[1]) : nothing
                    
    # initialize the optimizer cache
    optimizer_instance = enable_cuda ? Optimizer(CUDA.device(), o, model) : Optimizer(o, model)
                    
    @showprogress "Training network ..." for i in 1:training_steps
        # draw a mini batch 
        indices = Int.(ceil.(rand(batch_size)*num))
        x_batch = enable_cuda ? (train_x_reshaped[:, :, indices] |> cu) : train_x_reshaped[:, :, indices]
        y_batch = enable_cuda ? (train_y_encoded[:, indices] |> cu) : train_y_encoded[:, indices]
                    
        # compute the gradient using Zygote
        dp = gradient(ps -> loss(ps, x_batch, y_batch), ps)[1]
                    
        #update the cache of the optimizer and the parameter
        optimization_step!(optimizer_instance, model, ps, dp)    
                    
        # compute the loss at the current step
        loss_array[1+i] = give_training_error ? (enable_cuda ? loss(ps, train_x_reshaped |> cu, train_y_encoded |> cu) : loss(ps, train_x_reshaped, train_y_encoded)) : 0.
                    
    end
    #println("final loss: ", loss_array[end])
    enable_cuda ? println("final test loss: ", loss(ps, test_x_reshaped |> cu, test_y_encoded |> cu),"\n") : println("final test loss: ", loss(ps, test_x_reshaped, test_y_encoded),"\n")
                    
    loss_array
end

batch_size = 64
n_epochs = .1
o = AdamOptimizer(0.01f0, 0.9f0, 0.99f0, 1.0f-8)
enable_cuda = false
give_training_error = false

loss_arrays = NamedTuple{keys(models)}(Tuple(training(model, batch_size, n_epochs, o, enable_cuda, give_training_error) for model in models))

function plot_stuff()
    p = plot(loss_array₀, label="0")
    plot!(p, loss_array₁, label="1")
    plot!(p, loss_array₂, label="2")
    plot!(p, loss_array₃, label="3")
    plot!(p, loss_array₄, label="4")
end

give_training_error ? plot_stuff : nothing 