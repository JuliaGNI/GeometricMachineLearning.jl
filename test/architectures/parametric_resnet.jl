# `ParametricResNet` is the baseline a generalized Hamiltonian architecture is compared against: its
# blocks take the parameters of the system beside the state, and nothing in it makes the map
# symplectic. The expected values are the composition of the layers' closed forms, written out in
# `helpers/resnet_forward.jl`, over `params(nn)` read back one layer at a time.

using GeometricMachineLearning
using GeometricMachineLearning: ParametricResNet, ParametricResNetLayer, WideResNetLayer,
                                NullParameters, ParametricDataLoader, params,
                                parameterlength
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/resnet_forward.jl")

const DIM = 4
const WIDTH = 6

# the first entry of a `NamedTuple` of parameters comes first, and an array leaf of two entries
# follows it
const MU = (a = 0.5, b = [-1.5, 2.0])
const MU_VECTOR = [0.5, -1.5, 2.0]

# The forward pass of a `ParametricResNet` of two nonlinear blocks and the linear one, from the
# closed form of each layer.
function parametric_resnet_forward(x::AbstractArray, μ::AbstractVector, ps)
    y = parametric_forward(x, μ, ps[1], tanh)
    y = parametric_forward(y, μ, ps[2], tanh)
    parametric_forward(y, μ, ps[3], identity)
end

@testset "nn(x, μ) of a ParametricResNet is the composition of its layers ($T)" for T in REAL_ELTYPES
    arch = ParametricResNet(DIM; width = WIDTH, n_blocks = 2, parameters = MU)
    nn = NeuralNetwork(arch, T)
    μ = (a = T(MU.a), b = T.(MU.b))
    μv = T.(MU_VECTOR)

    # one sample as a vector, and as a `(q, p)` pair
    x = randn(Random.Xoshiro(1), T, DIM)
    expected = parametric_resnet_forward(x, μv, params(nn))
    output = @inferred nn(x, μ)
    @test eltype(output) === T
    @test isapprox(output, expected; atol = 10 * eps(T), rtol = 10 * eps(T))

    z = (q = randn(Random.Xoshiro(2), T, DIM ÷ 2), p = randn(Random.Xoshiro(3), T, DIM ÷ 2))
    expected = parametric_resnet_forward(vcat(z.q, z.p), μv, params(nn))
    output = @inferred nn(z, μ)
    @test keys(output) == (:q, :p)
    @test isapprox(vcat(output.q, output.p), expected;
        atol = 10 * eps(T), rtol = 10 * eps(T))
end

# Unlike `ResNet`, this architecture makes no choice on the width: every block is a
# `ParametricResNetLayer`, also where the width is the system dimension.
@testset "ParametricResNet keeps its own layer at a width equal to the dimension" begin
    for width in (DIM, WIDTH)
        layers = Chain(ParametricResNet(DIM; width = width, n_blocks = 2)).layers
        @test length(layers) == 3
        @test all(l -> l isa ParametricResNetLayer, layers)
    end
end

@testset "the DataLoader constructor keeps the system parameters (bug 5)" begin
    dl = DataLoader(rand(Random.Xoshiro(4), DIM, 20); suppress_info = true)
    arch = ParametricResNet(dl, 2, WIDTH; parameters = MU)

    @test arch.parameters == MU
    @test arch.sys_dim == DIM
    @test arch.n_blocks == 2
    @test arch.width == WIDTH

    # the parameters are what the extra width of every upscaling weight is made of: |μ| = 3 entries
    # in each of the two nonlinear blocks and in the linear one
    parameter_free = ParametricResNet(dl, 2, WIDTH)
    @test parameter_free.parameters isa NullParameters
    @test parameterlength(NeuralNetwork(arch)) -
          parameterlength(NeuralNetwork(parameter_free)) ==
          (2 + 1) * WIDTH * 3
end

@testset "train! of a ParametricResNet on a ParametricDataLoader ($T)" for T in REAL_ELTYPES
    m, nt = 3, 5
    data = [T(j + t) for _ in 1:2, t in 1:nt, j in 1:m]
    μs = [(a = T(j),) for j in 1:m]
    dl = ParametricDataLoader(data, μs)

    nn = NeuralNetwork(ParametricResNet(2; width = 3, n_blocks = 1, parameters = (a = 1.0,)), T)
    initial = deepcopy(params(nn))
    opt = TrainingOptimizer(nn.params; algorithm = Adam())
    history = train!(nn, opt, dl, Batch(4), 2; show_progress = false)

    @test length(history) == 2
    @test eltype(history) === T
    @test all(isfinite, history)
    # one step moves the parameters
    @test params(nn)[1].upscale_weight != initial[1].upscale_weight
end

@testset "the parametric ResNet names are public, not exported, and in the manual" begin
    for name in (:WideResNetLayer, :ParametricResNetLayer, :ParametricResNet)
        @test Base.ispublic(GeometricMachineLearning, name)
        @test !Base.isexported(GeometricMachineLearning, name)
    end

    page = read(
        joinpath(pkgdir(GeometricMachineLearning), "docs", "src", "architectures",
            "neural_network_integrators.md"),
        String)
    for name in ("WideResNetLayer", "ParametricResNetLayer", "ParametricResNet")
        @test occursin("GeometricMachineLearning.$name\n", page)
    end
end
