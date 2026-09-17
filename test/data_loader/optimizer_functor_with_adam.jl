import AbstractNeuralNetworks: AbstractExplicitLayer, Chain
using GeometricMachineLearning, Test
import Random

Random.seed!(1234)

# this function should be made part of AbstractNeuralNetworks !!!
function Chain(c::Chain, d::AbstractExplicitLayer)
    Chain(c.layers..., d)
end

"""
This creates a dummy classification data set: an input tensor already in the *time series* format the
transformer consumes and a one-hot encoded target.

`GMLDatasets` builds such a `DataLoader` from an image data set — that is what its `patch_length`
keyword is for. Here the tensors are random, because what is under test is the optimizer and not the
data, and this way the test needs neither `GMLDatasets` nor its image utilities.
"""
function create_dummy_classification_data(;
        T = Float32, input_dim = 9, n_patches = 4, n_images = 10, n_classes = 10)
    input = rand(T, input_dim, n_patches, n_images)
    output = zeros(Int, n_classes, 1, n_images)
    for (i, label) in pairs(rand(1:n_classes, n_images))
        output[label, 1, i] = 1
    end
    DataLoader{T, typeof(input), typeof(output), :TimeSeries}(
        input, output, input_dim, n_patches, n_images, n_classes, 1
    )
end

function test_optimization_with_adam(; T = Float32, input_dim = 9, n_patches = 4,
        n_images = 10, n_classes = 10, n_heads = 3)
    dl = create_dummy_classification_data(;
        T = T, input_dim = input_dim, n_patches = n_patches,
        n_images = n_images, n_classes = n_classes)

    # batch size is equal to two
    batch = Batch(2)

    # the transformer is called with `n_heads` heads and two layers
    model = Chain(Transformer(dl.input_dim, n_heads, 2; Stiefel = true),
        ClassificationLayer(dl.input_dim, dl.output_dim, σ))

    nn_obj = NeuralNetwork(model, CPU(), Float32)
    ps = nn_obj.params

    loss = FeedForwardLoss()

    loss₁ = loss(model, ps, dl.input, dl.output)

    opt = Optimizer(Adam(), nn_obj)
    λY = GlobalSection(ps)
    loss_average = optimize_for_one_epoch!(opt, model, ps, dl, batch, loss, λY)

    loss₃ = loss(model, ps, dl.input, dl.output)

    #check if the loss decreases during optimization
    @test loss₁ > loss_average > loss₃
end

test_optimization_with_adam()
