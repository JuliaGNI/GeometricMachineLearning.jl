using GeometricMachineLearning
using LinearAlgebra: det
using Zygote: jacobian
using Test
import Random

Random.seed!(1234)

function test_volume_preservation(layer::GeometricMachineLearning.AbstractExplicitLayer,
        ps::NamedTuple, b::AbstractVector{T}) where {T}
    jac_mat = jacobian(b -> layer(b, ps), b)[1]
    @test det(jac_mat) ≈ one(T)
end

function test_volume_preserving_feedforward(dim₁ = 5; T::Type = Float32)
    layer₁ = VolumePreservingLowerLayer(dim₁; use_bias = false)
    layer₂ = VolumePreservingLowerLayer(dim₁; use_bias = true)
    layer₃ = VolumePreservingUpperLayer(dim₁; use_bias = false)
    layer₄ = VolumePreservingUpperLayer(dim₁; use_bias = true)

    ps₁ = NeuralNetwork(Chain(layer₁), CPU(), T).params.L1
    ps₂ = NeuralNetwork(Chain(layer₂), CPU(), T).params.L1
    ps₃ = NeuralNetwork(Chain(layer₃), CPU(), T).params.L1
    ps₄ = NeuralNetwork(Chain(layer₄), CPU(), T).params.L1

    # test if application to matrix and tensor gives same result
    test_vector = rand(T, dim₁)
    test_volume_preservation(layer₁, ps₁, test_vector)
    test_volume_preservation(layer₂, ps₂, test_vector)
    test_volume_preservation(layer₃, ps₃, test_vector)
    test_volume_preservation(layer₄, ps₄, test_vector)
end

test_volume_preserving_feedforward(; T = Float32)
test_volume_preserving_feedforward(; T = Float64)

# A vector in, a vector out. Neither layer shapes the result: the layer without a bias is
# `x + d.activation.(ps.weight * x)`, and `StrictlyLowerTriangular * ::AbstractVector` returns a vector,
# which is what `LinearAlgebra` returns for every other matrix type.
l = VolumePreservingLowerLayer(2, identity; use_bias = false)
A = StrictlyLowerTriangular([1], 2)
@test l(ones(eltype(A), 2), (weight = A,)) == [1, 2]
l = VolumePreservingUpperLayer(2, identity; use_bias = false)
A = StrictlyUpperTriangular([1], 2)
@test l(ones(eltype(A), 2), (weight = A,)) == [2, 1]

# The output of both triangles, with and without a bias, pinned bit for bit against the formula
# written out with `LinearAlgebra`'s own product. The parameters are written out rather than drawn,
# so the pin is a property of the layer: it fails on a functor that drops the activation, the bias
# or the shift, and holds on every input rank the layer accepts.
const VPFF_STORAGE = [0.25, -0.5, 0.75, 1.0, -1.25, 0.5]
const VPFF_BIAS = [0.1, -0.2, 0.3, -0.4]
const VPFF_INPUT = [0.25, -0.5, 1.0, 0.75]

# The formula of the layer for one array, slice by slice on a tensor.
function vpff_reference(W, b, x)
    wx = W * x
    b === nothing ? x + tanh.(wx) : x + tanh.(wx .+ b)
end

@testset "the volume-preserving feedforward layers pin their output ($T)" for T in (
    Float32, Float64)
    S = T.(VPFF_STORAGE)
    b = T.(VPFF_BIAS)
    x = T.(VPFF_INPUT)
    X = repeat(x, 1, 3)
    XT = repeat(x, 1, 3, 2)

    for (layer, triangle) in (
            (VolumePreservingLowerLayer, StrictlyLowerTriangular),
            (VolumePreservingUpperLayer, StrictlyUpperTriangular)),
        use_bias in (false, true)

        d = layer(4, tanh; use_bias)
        weight = triangle(S, 4)
        W = Matrix(weight)
        ps = use_bias ? (weight = weight, bias = b) : (weight = weight,)
        bias_or_nothing = use_bias ? b : nothing

        y = d(x, ps)
        @test y == vpff_reference(W, bias_or_nothing, x)
        @test y isa AbstractVector{T}
        @test eltype(y) === T

        Y = d(X, ps)
        @test Y == vpff_reference(W, bias_or_nothing, X)
        @test eltype(Y) === T

        YT = d(XT, ps)
        reference = similar(XT)
        for k in axes(XT, 3)
            reference[:, :, k] = vpff_reference(W, bias_or_nothing, XT[:, :, k])
        end
        @test YT == reference
        @test eltype(YT) === T
    end
end
