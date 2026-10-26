# `WideResNetLayer` is `ResNetLayer` with a hidden layer of its own width: it upscales to `width`,
# applies the activation and downscales again, and adds its input. The expected value here is the
# closed form written out with `*` and `reshape`, so it does not go through the layer's own `_mul`,
# and the weights are drawn once and handed to the layer as they are.

using GeometricMachineLearning
using GeometricMachineLearning: WideResNetLayer, parameterlength
using AbstractNeuralNetworks: initialparameters, GlorotUniform
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/resnet_forward.jl")

const DIM = 4
const WIDTH = 7

# The number of entries `initialparameters` allocates for a layer: the length of every array of the
# tree it returns.
allocated_entries(x::AbstractArray) = length(x)
allocated_entries(x::NamedTuple) = sum(allocated_entries, values(x); init = 0)

function fixed_parameters(::Type{T}) where {T}
    rng = Random.Xoshiro(7)
    (upscale_weight = randn(rng, T, WIDTH, DIM),
        downscale_weight = randn(rng, T, DIM, WIDTH),
        upscale_bias = randn(rng, T, WIDTH),
        bias = randn(rng, T, DIM))
end

@testset "parameterlength(::WideResNetLayer) counts the entries initialparameters allocates" begin
    l = WideResNetLayer(DIM, WIDTH, tanh)
    ps = initialparameters(Random.Xoshiro(1), GlorotUniform(), l, CPU(), Float64)
    @test parameterlength(l) == allocated_entries(ps)
    @test parameterlength(l) == WIDTH * (DIM + 1) + DIM * (WIDTH + 1)
    # the shapes are the ones the forward pass needs, so a layer drawn by `initialparameters` is
    # applicable as it is
    @test size(l(randn(Random.Xoshiro(2), DIM), ps)) == (DIM,)
    @test size(l(randn(Random.Xoshiro(3), DIM, 3), ps)) == (DIM, 3)
end

@testset "the forward pass of WideResNetLayer is the closed form ($T)" for T in REAL_ELTYPES
    σ = tanh
    l = WideResNetLayer(DIM, WIDTH, σ)
    ps = fixed_parameters(T)

    for input in (randn(Random.Xoshiro(2), T, DIM),
        randn(Random.Xoshiro(3), T, DIM, 3),
        randn(Random.Xoshiro(4), T, DIM, 3, 2))
        output = l(input, ps)
        @test size(output) == size(input)
        @test eltype(output) === T
        @test isapprox(output, wide_resnet_forward(input, ps, σ);
            atol = 10 * eps(T), rtol = 10 * eps(T))
    end

    # a `(q, p)` input is the stacked vector, split again
    z = (q = randn(Random.Xoshiro(5), T, DIM ÷ 2), p = randn(Random.Xoshiro(6), T, DIM ÷ 2))
    expected = wide_resnet_forward(vcat(z.q, z.p), ps, σ)
    output = l(z, ps)
    @test keys(output) == (:q, :p)
    @test isapprox(vcat(output.q, output.p), expected;
        atol = 10 * eps(T), rtol = 10 * eps(T))
end

@testset "WideResNetLayer is public and not exported" begin
    @test Base.ispublic(GeometricMachineLearning, :WideResNetLayer)
    @test !Base.isexported(GeometricMachineLearning, :WideResNetLayer)
end
