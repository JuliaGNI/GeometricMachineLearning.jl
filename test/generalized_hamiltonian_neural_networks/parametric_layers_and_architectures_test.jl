# Construction and a forward pass for each parameter-dependent and forced piece. The other PGHNN
# tests build a plain `GeneralizedHamiltonianArchitecture`, so none of these were evaluated
# anywhere -- which is how `ForcedGeneralizedHamiltonianArchitecture`, exported and documented,
# came to have a forward pass that threw.

using GeometricMachineLearning
using GeometricMachineLearning: ForcingLayerQ, ForcingLayerP, ForcingLayerQP,
                                ParametricResNetLayer, WideResNetLayer, ParametricResNet
using AbstractNeuralNetworks: params
using Random: seed!
using Test

seed!(1234)

const DIM = 4
const HALF = DIM ÷ 2
const WIDTH = 8
const SYSTEM_PARAMETERS = (m = 1.0, ω = π / 2)

finite(x::AbstractArray) = all(isfinite, x)
finite(qp::NamedTuple) = finite(qp.q) && finite(qp.p)

# The `Q`/`P`/`QP` suffix names what the forcing *depends on*, not what it changes: a force enters
# the `ṗ` equation, so all three add to `p` and leave `q` alone.
@testset "ForcingLayer$name" for (name, Layer, depends_on) in (
        ("Q", ForcingLayerQ, (:q,)), ("P", ForcingLayerP, (:p,)), ("QP", ForcingLayerQP, (:q, :p)))
    layer = Layer(DIM; parameters = SYSTEM_PARAMETERS)
    nn = NeuralNetwork(layer)
    @test parameterlength(nn) > 0

    z = (q = rand(HALF), p = rand(HALF))
    out = layer(z, SYSTEM_PARAMETERS, params(nn))
    @test keys(out) == (:q, :p)
    @test size(out.q) == size(z.q) && size(out.p) == size(z.p)
    @test finite(out)
    @test out.q == z.q
    @test out.p != z.p

    # perturb one coordinate at a time: the forcing may only move when a coordinate it is named
    # after does
    for coordinate in (:q, :p)
        perturbed = merge(z, NamedTuple{(coordinate,)}((z[coordinate] .+ 1.0,)))
        moved = layer(perturbed, SYSTEM_PARAMETERS, params(nn)).p .- perturbed.p !=
                out.p .- z.p
        @test moved == (coordinate in depends_on)
    end

    # the same layer applied to the concatenated array form
    array_out = layer(vcat(z.q, z.p), SYSTEM_PARAMETERS, params(nn))
    @test array_out ≈ vcat(out.q, out.p)
end

@testset "WideResNetLayer" begin
    layer = WideResNetLayer(DIM, WIDTH, tanh)
    nn = NeuralNetwork(Chain(layer))
    ps = params(nn).L1
    @test parameterlength(layer) == WIDTH * (DIM + 1) + DIM * (WIDTH + 1)

    for input in (rand(DIM), rand(DIM, 3), rand(DIM, 3, 2))
        out = layer(input, ps)
        @test size(out) == size(input)
        @test finite(out)
    end

    z = (q = rand(HALF), p = rand(HALF))
    out = layer(z, ps)
    @test keys(out) == (:q, :p)
    @test out ≈ (q = layer(vcat(z.q, z.p), ps)[1:HALF], p = layer(vcat(z.q, z.p), ps)[(HALF + 1):DIM])
end

@testset "ParametricResNetLayer" begin
    layer = ParametricResNetLayer(DIM, WIDTH, tanh;
                                  parameters = SYSTEM_PARAMETERS, return_parameters = false)
    nn = NeuralNetwork(Chain(layer))
    ps = params(nn).L1

    # one `NamedTuple` of system parameters describes one sample, so a matrix input is a single
    # column; a batch is a vector of parameter sets, which is what `ParametricResNet` builds
    for input in (rand(DIM), rand(DIM, 1))
        out = layer(input, SYSTEM_PARAMETERS, ps)
        @test size(out) == size(input)
        @test finite(out)
    end
    @test_throws AssertionError layer(rand(DIM, 3), SYSTEM_PARAMETERS, ps)

    z = (q = rand(HALF), p = rand(HALF))
    @test finite(layer(z, SYSTEM_PARAMETERS, ps))
end

@testset "ResNet with a width of its own" begin
    # `sys_dim == width` keeps the plain `ResNetLayer`; a different width switches to
    # `WideResNetLayer`, which is the path `ParametricResNet` compares against
    narrow = NeuralNetwork(ResNet(DIM, 2, DIM))
    wide = NeuralNetwork(ResNet(DIM, 2, WIDTH))
    @test parameterlength(wide) > parameterlength(narrow)
    @test size(wide(rand(DIM))) == (DIM,)
    @test finite(wide(rand(DIM)))
end

@testset "ParametricResNet" begin
    arch = ParametricResNet(DIM; width = WIDTH, n_blocks = 2, parameters = SYSTEM_PARAMETERS)
    nn = NeuralNetwork(arch)
    out = nn.model(rand(DIM), SYSTEM_PARAMETERS, params(nn))
    @test size(out) == (DIM,)
    @test finite(out)

    # the `DataLoader` constructor used to accept `parameters` and drop it
    dl = DataLoader(rand(DIM, 20); suppress_info = true)
    @test ParametricResNet(dl, 2, WIDTH; parameters = SYSTEM_PARAMETERS).parameters ==
          SYSTEM_PARAMETERS
end

@testset "ForcedSympNet $forcing_type" for forcing_type in (:Q, :P, :QP)
    nn = NeuralNetwork(ForcedSympNet(DIM; forcing_type = forcing_type))
    out = nn(rand(DIM))
    @test size(out) == (DIM,)
    @test finite(out)
end

@testset "ForcedGeneralizedHamiltonianArchitecture $forcing_type" for forcing_type in (:Q, :P, :QP)
    arch = ForcedGeneralizedHamiltonianArchitecture(DIM; parameters = SYSTEM_PARAMETERS,
                                                    forcing_type = forcing_type)
    nn = NeuralNetwork(arch)
    out = nn(rand(DIM), SYSTEM_PARAMETERS)
    @test size(out) == (DIM,)
    @test finite(out)
end

@test_throws ErrorException ForcedGeneralizedHamiltonianArchitecture(DIM; forcing_type = :X)
