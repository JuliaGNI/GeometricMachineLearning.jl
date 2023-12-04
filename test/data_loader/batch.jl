using AbstractNeuralNetworks: AbstractExplicitLayer, Chain
using GeometricMachineLearning, Test

"""
This creates a dummy MNIST data set.

TODO: include tests to check if all elements are batched!
"""
function create_dummy_mnist(;T=Float32, dim₁=6, dim₂=6, n_images=10)
    rand(T, dim₁, dim₂, n_images), Int.(floor.(10*rand(T, n_images)))
end

dl = DataLoader(create_dummy_mnist()...; patch_length=3)
# batch size is equal to two
batch = Batch(2)

# this function should be made part of AbstractNeuralNetworks !!!
function Chain(c::Chain, d::AbstractExplicitLayer) 
    Chain(c.layers..., d)
end

# input dim is 3^2 = 9
model = Chain(Transformer(dl.input_dim, 3, 2; Stiefel=true), Classification(dl.input_dim, 10, σ))
ps = initialparameters(CPU(), Float32, model)

loss₁ = GeometricMachineLearning.loss(model, ps, dl)

opt = Optimizer(AdamOptimizer(), ps)
loss_average = optimize_for_one_epoch!(opt, model, ps, dl, batch)

loss₃ = GeometricMachineLearning.loss(model, ps, dl)

@test loss₁ > loss_average > loss₃