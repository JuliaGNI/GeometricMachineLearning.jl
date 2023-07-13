using GeometricMachineLearning, LinearAlgebra, ProgressMeter
import Zygote, Random, MLDatasets, Flux 

using GeometricMachineLearning, LinearAlgebra, ProgressMeter, CUDA #, Plots
import Lux, Zygote, Random, MLDatasets, Flux

image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 16
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

train_x_reshaped = zeros(Float32, patch_length^2, patch_number, size(train_x, 3))
test_x_reshaped = zeros(Float32, patch_length^2, patch_number, size(test_x, 3))

# preprocessing steps (also perform rescaling so that the images have values between 0 and 1)
for i in axes(train_x, 3)
    train_x_reshaped[:, :, i] = split_and_flatten(train_x[:, :, i], patch_length)/255
end
for i in axes(test_x, 3)
    test_x_reshaped[:, :, i] = split_and_flatten(test_x[:, :, i], patch_length)/255
end

# implement this encoding yourself!
train_y = Flux.onehotbatch(train_y, 0:9) 
test_y = Flux.onehotbatch(test_y, 0:9)

model₀ = Lux.Chain(Lux.Dense(49, 49, tanh), Classification(patch_length^2, 10, use_bias=false))
model₁ = Lux.Chain( MultiHeadAttention(patch_length^2, n_heads, add_connection=false, Stiefel=false),
                    MultiHeadAttention(patch_length^2, n_heads, add_connection=false, Stiefel=false),
                    Classification(patch_length^2, 10, use_bias=false))
model₂ = Lux.Chain(MultiHeadAttention(patch_length^2, n_heads, add_connection=true, Stiefel=false),
                    MultiHeadAttention(patch_length^2, n_heads, add_connection=true, Stiefel=false),
                    Classification(patch_length^2, 10, use_bias=false))
model₃ = Lux.Chain(MultiHeadAttention(patch_length^2, n_heads, add_connection=false, Stiefel=true),
                    MultiHeadAttention(patch_length^2, n_heads, add_connection=false, Stiefel=true),
                    Classification(patch_length^2, 10, use_bias=false))
model₄ = Lux.Chain(MultiHeadAttention(patch_length^2, n_heads, add_connection=true, Stiefel=true),
                    MultiHeadAttention(patch_length^2, n_heads, add_connection=true, Stiefel=true),
                    Classification(patch_length^2, 10, use_bias=false))

const num = 60000
function training(model::Lux.Chain, batch_size=128, n_epochs=1, o=AdamOptimizer())
    ps, st = Lux.setup(CUDA.device(), Random.default_rng(), model)

    function loss(ps, x, y)
        x_eval = Lux.apply(model, x |> cu, ps, st)[1]
        norm(x_eval - (y |> cu))/sqrt(size(y, 2))
    end

    training_steps = Int(ceil(n_epochs*num/batch_size))
    loss_array = zeros(training_steps + 1)
    loss_array[1] = loss(ps, train_x_reshaped |> cu, train_y |> cu)
    println("initial loss: ", loss_array[1])

    optimizer_instance = Optimizer(CUDA.device(), o, model)

    @showprogress "Training network ..." for i in 1:training_steps
        indices = Int.(ceil.(rand(batch_size)*num))
        x_batch = train_x_reshaped[:, :, indices] |> cu 
        y_batch = train_y[:, indices] |> cu

        dp = Zygote.gradient(ps -> loss(ps, x_batch, y_batch), ps)[1]

        optimization_step!(optimizer_instance, model, ps, dp)    
        loss_array[1+i] = loss(ps, train_x_reshaped |> cu, train_y |> cu)
    end
    println("final loss: ", loss_array[end])
    println("final test loss: ", loss(ps, test_x_reshaped |> cu, test_y |> cu),"\n")

    loss_array
end

loss_array₀ = training(model₀)
loss_array₁ = training(model₁)
loss_array₂ = training(model₂)
loss_array₃ = training(model₃)
loss_array₄ = training(model₄)