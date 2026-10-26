# `accuracy` is documented API -- `docs/src/data_loader/data_loader.md` renders both its methods --
# and no other function of the package and no script calls it. A documented function with no
# caller and no test is a function nobody notices breaking, so this file exercises it.
#
# The loader comes from `DataLoader(input, output)`. The output's element type is the input's
# or an integer, so a real input and an integer one-hot target are one call; any other pair of
# element types is a `MethodError`.

using GeometricMachineLearning
using Random
using Test
using Zygote

Random.seed!(1234)

const input_dim = 4
const seq_length = 3
const n_classes = 3
const n_params = 6

const input = rand(Float32, input_dim, seq_length, n_params)

# One `DataLoader` per label assignment and element type, over the same input. A label of `0` marks a
# sample that carries none: its one-hot column stays zero, so the target differs from the prediction
# in one entry where a wrong label differs in two.
function classification_loader(classes::AbstractVector{<:Integer}, ::Type{T} = Float32) where {T}
    output = zeros(Int64, n_classes, 1, n_params)
    for (k, c) in pairs(classes)
        iszero(c) && continue
        output[c, 1, k] = 1
    end
    DataLoader(T.(input), output; suppress_info = true)
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

# The fraction is read in the element type of the loader's input, so a `Float32` run of the model
# gives a `Float32` measure and does not silently widen at the last step. A loader of the model's own
# element type cannot tell the loader's type from the model's output's; the mixed pair below does.
@testset "the accuracy has the element type of the loader's input ($T)" for T in (Float32, Float64)
    dl = classification_loader(predicted, T)
    @test eltype(dl) === T
    @test GeometricMachineLearning.accuracy(model, ps, dl) isa T
    @test GeometricMachineLearning.accuracy(model, ps, dl) == one(T)

    wrong = classification_loader(rotated, T)
    @test GeometricMachineLearning.accuracy(model, ps, wrong) isa T
    @test iszero(GeometricMachineLearning.accuracy(model, ps, wrong))
end

# The loader's element type decides the result, not the element type of the model or of its output: a
# `Float64` model on a `Float32` loader gives a `Float32`. The two networks do not agree on every
# label, so the mixed case labels the loader from the `Float64` network's own predictions.
@testset "a Float64 model on a Float32 loader still gives a Float32" begin
    nn64 = NeuralNetwork(model, Float64)
    ps64 = GeometricMachineLearning.params(nn64)
    dl32 = classification_loader(predicted, Float32)

    # the output this loader makes is `Float64` here, so the two element types differ
    estimate64 = GeometricMachineLearning.assign_output_estimate(nn64.model(dl32.input, ps64), 1)
    @test eltype(estimate64) === Float64

    @test GeometricMachineLearning.accuracy(nn64, dl32) isa Float32
    @test GeometricMachineLearning.accuracy(nn64.model, ps64, dl32) isa Float32

    # labelled with what the `Float64` network predicts, every label is correct
    predicted64 = [argmax(estimate64[:, 1, k]) for k in 1:n_params]
    @test GeometricMachineLearning.accuracy(nn64,
        classification_loader(predicted64, Float32)) == one(Float32)

    @test GeometricMachineLearning.accuracy(nn64,
        classification_loader(predicted, Float64)) isa Float64
end

# A sample that carries no label is half an error: the difference between the target and the
# prediction is 1 in its one-hot column, where a wrong label gives 2, and the count is halved rather
# than floored. A floored count drops that sample and reports every label as correct.
@testset "a sample with no label halves the error count ($T)" for T in (Float32, Float64)
    classes = copy(predicted)
    classes[1] = 0
    a = GeometricMachineLearning.accuracy(model, ps, classification_loader(classes, T))
    @test a isa T
    @test a == (T(n_params) - T(1) / 2) / T(n_params)
    @test a < one(T)
end

@testset "the NeuralNetwork method agrees with the Chain method" begin
    for classes in (predicted, rotated)
        dl = classification_loader(classes)
        @test GeometricMachineLearning.accuracy(nn, dl) ==
              GeometricMachineLearning.accuracy(model, ps, dl)
    end
end

# The crop of `assign_output_estimate` is the last `prediction_window` columns, and its gradient is
# Zygote's `getindex` adjoint — zero on the cropped columns, and a cotangent of the primal's element
# type whatever the seed's.
@testset "assign_output_estimate crops the last columns ($T)" for T in (Float32, Float64)
    x = T.(reshape(1:(3 * 5 * 4), 3, 5, 4))
    @test GeometricMachineLearning.assign_output_estimate(x, 2) == x[:, 4:5, :]
    @test GeometricMachineLearning.assign_output_estimate(x, 5) == x
    @test eltype(GeometricMachineLearning.assign_output_estimate(x, 2)) === T
end

@testset "the gradient of assign_output_estimate is Zygote's ($T)" for T in (Float32, Float64)
    x = T.(reshape(1:(3 * 5 * 4), 3, 5, 4))
    crop(z) = GeometricMachineLearning.assign_output_estimate(z, 2)
    gradient = only(Zygote.gradient(z -> sum(abs2, crop(z)), x))
    @test all(iszero, gradient[:, 1:3, :])
    @test gradient[:, 4:5, :] == 2 .* x[:, 4:5, :]
    _, back = Zygote.pullback(crop, x)
    @test eltype(only(back(ones(Float64, 3, 2, 4)))) === T
end
