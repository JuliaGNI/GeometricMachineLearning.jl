"""
Short script for comparative study of transformer and Stiefel-transformer on MNIST. CUDA version.
"""

using GeometricMachineLearning, LinearAlgebra, ProgressMeter, CUDA #, Plots
import Lux, Zygote, Random, MLDatasets, Flux

image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 8
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

train_x_reshaped = zeros(Float32, patch_length^2, patch_number, size(train_x, 3))
test_x_reshaped = zeros(Float32, patch_length^2, patch_number, size(test_x, 3))

# preprocessing steps (also perform rescaling so that the images have values between 0 and 1)
for i in 1:size(train_x, 3)
    train_x_reshaped[:, :, i] = split_and_flatten(train_x[:, :, i], patch_length)/255
end
for i in 1:size(test_x, 3)
    test_x_reshaped[:, :, i] = split_and_flatten(test_x[:, :, i], patch_length)/255
end

# implement this encoding yourself!
train_y = Flux.onehotbatch(train_y, 0:9) 
test_y = Flux.onehotbatch(test_y, 0:9)

# encoder layer - final layer has to be added for evaluation purposes!
Ψᵉ₁ = Lux.Chain(
    #Embedding(patch_length^2, patch_number),
    Transformer(patch_length^2, n_heads, n_layers, add_connection=false, Stiefel=false),
    Classification(patch_length^2, 10, use_bias=false)
)

Ψᵉ₂ = Lux.Chain(
    # Embedding(patch_length^2, patch_number),
    Transformer(patch_length^2, n_heads, n_layers, add_connection=false, Stiefel=true),
    Classification(patch_length^2, 10, use_bias=false)
)

# err_freq is the frequency with which the error is computed (e.g. every 100 steps)
function transformer_training(Ψᵉ::Lux.Chain, batch_size=64, n_epochs=10, o=AdamOptimizer())
    ps, st = Lux.setup(CUDA.device(), Random.default_rng(), Ψᵉ) 

    steps_in_one_epoch = Int(ceil(60000/batch_size))
    training_steps = n_epochs*steps_in_one_epoch
    # loss_sing
    function loss(ps, x, y)
        x_eval = Lux.apply(Ψᵉ, x |> cu, ps, st)[1]
        #compute the norm of the missclassification divided by the squre root of the batch size
        norm(x_eval - (y |> cu))/sqrt(size(y, 2))
    end

    num = size(train_x_reshaped, 3)

    loss_array = zeros(training_steps÷n_epochs + 1)
    loss_array[1] = loss(ps, train_x_reshaped |> cu, train_y |> cu)
    println("initial loss: ", loss_array[1])

    optimizer_instance = Optimizer(CUDA.device(), o, Ψᵉ)

    @showprogress "Training network ..." for i in 1:training_steps
        indices = Int.(ceil.(rand(batch_size)*num))
        x_batch = train_x_reshaped[:, :, indices] |> cu 
        y_batch = train_y[:, indices] |> cu

        dp = Zygote.gradient(ps -> loss(ps, x_batch, y_batch), ps)[1]

        optimization_step!(optimizer_instance, Ψᵉ, ps, dp)    
        if i%steps_in_one_epoch == 0
            loss_array[1+i÷steps_in_one_epoch] = loss(ps, train_x_reshaped |> cu, train_y |> cu)
            println("Current loss is: ",loss_array[1+i÷steps_in_one_epoch])
        end
    end
    println("final loss: ", loss_array[end])
    println("final test loss: ", loss(ps, test_x_reshaped |> cu, test_y |> cu),"\n")

    loss_array
end

batch_size = 512
n_epochs = 100
o = AdamOptimizer()

loss_array₁ = transformer_training(Ψᵉ₁, batch_size, n_epochs, o)
loss_array₂ = transformer_training(Ψᵉ₂, batch_size, n_epochs, o)

#=
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
=#