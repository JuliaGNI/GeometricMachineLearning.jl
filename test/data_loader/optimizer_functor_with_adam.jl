using AbstractNeuralNetworks: AbstractExplicitLayer, Chain
using GeometricMachineLearning, Test
import Random 

Random.seed!(1234)

# this function should be made part of AbstractNeuralNetworks !!!
function Chain(c::Chain, d::AbstractExplicitLayer) 
    Chain(c.layers..., d)
end

"""
This creates a dummy MNIST data set; i.e. its output are two tensors that look similarly to the ones in the MNIST data set.
"""
function create_dummy_mnist(; T=Float32, dim₁=6, dim₂=6, n_images=10)
    rand(T, dim₁, dim₂, n_images), Int.(floor.(10*rand(T, n_images)))
end

function test_optimization_with_adam(;T=Float32, dim₁=6, dim₂=6, n_images=10, patch_length=3)
    dl = DataLoader(create_dummy_mnist(T=T, dim₁=dim₁, dim₂=dim₂, n_images=n_images)...; patch_length=patch_length)
    
    # batch size is equal to two
    batch = Batch(2)

    # input dim is dim₁ / patch_length * dim₂ / pach_length; the transformer is called with dim₁ / patch_length and two layers
    model = Chain(Transformer(dl.input_dim, patch_length, 2; Stiefel=true), ClassificationLayer(dl.input_dim, 10, σ))

    ps = NeuralNetwork(model, CPU(), Float32).params

    loss = FeedForwardLoss()

    loss₁ = loss(model, ps, dl.input, dl.output)

    opt = Optimizer(AdamOptimizer(), ps)
    λY = GlobalSection(ps)
    loss_average = optimize_for_one_epoch!(opt, model, ps, dl, batch, loss, λY)

    loss₃ = loss(model, ps, dl.input, dl.output)

    #check if the loss decreases during optimization
    @test loss₁ > loss_average > loss₃
end

test_optimization_with_adam()