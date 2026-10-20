using GeometricMachineLearning, Test
using GeometricMachineLearning: ResNetLayer, params
import Random

include("../helpers/eltypes.jl")

@testset "ResNetLayer encodes its bias as VPFF does (use_bias = $use_bias)" for use_bias in (
    true, false)
    resnet = typeof(ResNetLayer(3; use_bias))
    vpff = typeof(VolumePreservingLowerLayer(3; use_bias))
    @test typeof(resnet.parameters[3]) === typeof(vpff.parameters[3])
    @test resnet.parameters[3] === vpff.parameters[3]
end

@testset "ResNetLayer type parameters" begin
    @test ResNetLayer(3) isa ResNetLayer{3, 3, :bias, typeof(identity)}
    @test ResNetLayer(3, tanh; use_bias = false) isa
          ResNetLayer{3, 3, :no_bias, typeof(tanh)}
end

@testset "ResNetLayer with and without a bias ($T)" for T in REAL_ELTYPES
    @test parameterlength(ResNetLayer(3; use_bias = false)) == 9
    @test parameterlength(ResNetLayer(3; use_bias = true)) == 12

    d = ResNetLayer(3, tanh; use_bias = false)
    ps = params(NeuralNetwork(Chain(d), CPU(), T; rng = Random.Xoshiro(3))).L1
    @test keys(ps) == (:weight,)
    @test length(ps.weight) == 9
    x = rand(Random.Xoshiro(4), T, 3, 5)
    y = d(x, ps)
    @test eltype(y) === T
    @test y == x + tanh.(ps.weight * x)

    b = ResNetLayer(3, tanh; use_bias = true)
    psb = params(NeuralNetwork(Chain(b), CPU(), T; rng = Random.Xoshiro(3))).L1
    @test keys(psb) == (:weight, :bias)
    @test length(psb.weight) + length(psb.bias) == 12
    # the bias is initialised to zero; a nonzero one shows that the forward pass adds it
    @test iszero(psb.bias)
    psb.bias .= T[1, -2, 3]
    yb = b(x, psb)
    @test eltype(yb) === T
    @test yb == x + tanh.(psb.weight * x .+ psb.bias)
end
