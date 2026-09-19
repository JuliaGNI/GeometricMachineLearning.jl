using GeometricMachineLearning
using Random
using Test

# `accuracy` is documented API -- `docs/src/data_loader/data_loader.md` renders both its methods --
# and nothing in the package, the tests or the scripts called it. A documented function with no
# caller and no test is a function nobody notices breaking, so this file exercises it rather than
# the docs entry being dropped.
#
# It also records what the audit found on the way: **no two-argument `DataLoader` constructor
# produces the type this function dispatches on.** The signature is
# `DataLoader{T, AT <: AbstractArray{T}, BT <: AbstractArray{T1}}` with `T1 <: Integer`, and every
# `DataLoader(input, output)` method requires both arrays to share one element type. A classifier's
# data are exactly the case where they differ: real inputs, integer one-hot targets. So the parametric
# constructor below is not a shortcut past a public one -- it is the only way in.

Random.seed!(1234)

const input_dim = 4
const seq_length = 3
const n_classes = 3
const n_params = 6

const input = rand(Float32, input_dim, seq_length, n_params)

# One `DataLoader` per label assignment, over the same input.
function classification_loader(classes::AbstractVector{<:Integer})
    output = zeros(Int64, n_classes, 1, n_params)
    for (k, c) in pairs(classes)
        output[c, 1, k] = 1
    end
    GeometricMachineLearning.DataLoader{
        Float32, Array{Float32, 3}, Array{Int64, 3}, :TimeSeries}(
        input, output, input_dim, seq_length, n_params, n_classes, 1)
end

model = Chain(Dense(input_dim, n_classes, tanh))
nn = NeuralNetwork(model, Float32)
ps = GeometricMachineLearning.params(nn)

# The labels are read off the network rather than drawn, so the two assertions below are exact and
# do not depend on the seed: label every sample with what this network already predicts and the
# accuracy is 1, rotate every label one class on and it is 0.
output_estimate = GeometricMachineLearning.assign_output_estimate(model(input, ps), 1)
predicted = [argmax(output_estimate[:, 1, k]) for k in 1:n_params]
rotated = [mod1(c + 1, n_classes) for c in predicted]

@testset "accuracy is 1 when every label is the predicted class, and 0 when none is" begin
    @test GeometricMachineLearning.accuracy(model, ps, classification_loader(predicted)) ==
          1
    @test GeometricMachineLearning.accuracy(model, ps, classification_loader(rotated)) == 0
end

@testset "the NeuralNetwork method agrees with the Chain method" begin
    for classes in (predicted, rotated)
        dl = classification_loader(classes)
        @test GeometricMachineLearning.accuracy(nn, dl) ==
              GeometricMachineLearning.accuracy(model, ps, dl)
    end
end
