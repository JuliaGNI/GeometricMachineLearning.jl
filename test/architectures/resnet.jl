# `ResNet` switches its block type on the width: a width equal to the system dimension keeps the
# plain `ResNetLayer`, and any other width builds a `WideResNetLayer`. `architecture_pins.jl` covers
# the first case through the recorded outputs, so this file covers the switch itself.

using GeometricMachineLearning
using GeometricMachineLearning: ResNetLayer, WideResNetLayer, parameterlength, params
using AbstractNeuralNetworks: Chain
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/resnet_forward.jl")

# Two wide blocks and the linear output layer `x ↦ x + (W x .+ b)`, from the closed form of each.
function wide_resnet_chain_forward(x::AbstractVector, ps)
    y = wide_resnet_forward(x, ps[1], tanh)
    y = wide_resnet_forward(y, ps[2], tanh)
    y + (ps[3].weight * y .+ ps[3].bias)
end

@testset "ResNet builds a WideResNetLayer for a width of its own" begin
    narrow = ResNet(4, 2, tanh)
    wide = ResNet(4, 2, 6, tanh)

    @test narrow.width == 4
    @test wide.width == 6

    @test all(l -> l isa ResNetLayer, Chain(narrow).layers)
    layers = Chain(wide).layers
    @test all(l -> l isa WideResNetLayer, layers[1:2])
    # the output layer applies the identity and keeps the system dimension in both cases
    @test layers[3] isa ResNetLayer
    @test Chain(narrow).layers[3] isa ResNetLayer
end

@testset "ResNet takes every Integer it documents" begin
    @test ResNet(Int32(4), 2, tanh).width == 4
    @test ResNet(Int32(4), Int32(2), Int32(6), tanh).width == 6
end

@testset "the DataLoader constructor carries the width" begin
    dl = DataLoader(rand(Random.Xoshiro(1), 4, 20); suppress_info = true)
    @test ResNet(dl, 2).width == 4
    @test ResNet(dl, 2, 6).width == 6
    @test all(l -> l isa ResNetLayer, Chain(ResNet(dl, 2)).layers[1:2])
    @test all(l -> l isa WideResNetLayer, Chain(ResNet(dl, 2, 6)).layers[1:2])
end

@testset "the forward pass of a wide ResNet ($T)" for T in REAL_ELTYPES
    wide = NeuralNetwork(ResNet(4, 2, 6, tanh), T)
    narrow = NeuralNetwork(ResNet(4, 2, tanh), T)
    @test parameterlength(wide) > parameterlength(narrow)

    x = randn(Random.Xoshiro(2), T, 4)
    y = wide(x)
    @test size(y) == (4,)
    @test eltype(y) === T
    @test isapprox(y, wide_resnet_chain_forward(x, params(wide));
        atol = 10 * eps(T), rtol = 10 * eps(T))

    q, p = randn(Random.Xoshiro(3), T, 2), randn(Random.Xoshiro(4), T, 2)
    z = wide((q = q, p = p))
    @test keys(z) == (:q, :p)
    @test isapprox(vcat(z.q, z.p), wide_resnet_chain_forward(vcat(q, p), params(wide));
        atol = 10 * eps(T), rtol = 10 * eps(T))
end
