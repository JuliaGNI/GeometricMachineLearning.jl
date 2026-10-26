# A `ForcedSympNet` is the layer sequence of a `GSympNet` with a forcing layer in front of every
# layer that changes `p`. The sequence is checked against the loop index and the type of each layer,
# the output against the gradient-only chain with the forcing parameters zeroed, and the training
# against the parameters of a forcing layer, which only move if the gradient reaches the nested
# `Chain` a forcing layer holds.

using GeometricMachineLearning
using GeometricMachineLearning: ForcedSympNet, ForcingLayer, params
using NeuralNetworkParameters: NetworkParameters, foldparameters, foreachparameters
using Test
import Random

include("../helpers/eltypes.jl")

const DIM = 4
const HALF = DIM ÷ 2
const UPSCALING = 6

# the indices of the forcing layers of a forced SympNet
forcing_indices(arch) = findall(l -> l isa ForcingLayer, collect(Chain(arch).layers))

arch_forcing_type(::ForcedSympNet{FT}) where {FT} = FT

# The layer sequence of the loop index, written out by hand: `q` is a `GradientLayerQ`, `p` a
# `GradientLayerP`, and `f` the forcing layer in front of every step that changes `p`. With
# `init_upper` the odd steps change `q`; without it the even ones do.
const FORCED_SEQUENCES = Dict(
    (true, 2) => (:q, :f, :p),
    (true, 3) => (:q, :f, :p, :q),
    (true, 4) => (:q, :f, :p, :q, :f, :p),
    (false, 2) => (:f, :p, :q),
    (false, 3) => (:f, :p, :q, :f, :p),
    (false, 4) => (:f, :p, :q, :f, :p, :q))

function layer_of(symbol::Symbol, arch)
    symbol === :q &&
        return GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.activation)
    symbol === :p &&
        return GradientLayerP(arch.dim, arch.upscaling_dimension, arch.activation)
    ForcingLayer(
        arch.dim, arch.upscaling_dimension, arch.n_forcing_layers, arch.activation;
        type = arch_forcing_type(arch))
end

@testset "the Chain of a ForcedSympNet alternates as the loop index says" begin
    for (init_upper, n_layers) in sort(collect(keys(FORCED_SEQUENCES))),
        forcing_type in (:Q, :P, :QP)

        arch = ForcedSympNet(DIM; n_layers = n_layers, init_upper = init_upper,
            upscaling_dimension = UPSCALING, forcing_type = forcing_type, n_forcing_layers = 3)
        layers = Chain(arch).layers
        sequence = FORCED_SEQUENCES[(init_upper, n_layers)]

        # `n_layers` gradient layers and one forcing layer per step that changes `p`
        @test length(sequence) == n_layers + count(==(:f), sequence)
        @test length(layers) == length(sequence)
        @test typeof.(layers) == map(symbol -> typeof(layer_of(symbol, arch)), sequence)

        # the forcing layers carry the type and the size the architecture gives them
        for index in forcing_indices(arch)
            @test layers[index] isa ForcingLayer{DIM, DIM, forcing_type, false}
            @test layers[index].width == UPSCALING
            @test layers[index].nhidden == 3
        end
        @test length(forcing_indices(arch)) == count(==(:f), sequence)
    end
end

# With every forcing parameter zero the forcing term is zero, and the network is the chain of its
# gradient layers alone.
@testset "a ForcedSympNet without its forcing is the chain of its gradient layers ($T)" for T in REAL_ELTYPES
    Random.seed!(5)
    arch = ForcedSympNet(DIM; n_layers = 4, upscaling_dimension = UPSCALING, forcing_type = :QP)
    nn = NeuralNetwork(arch, CPU(), T)
    layers = Chain(arch).layers

    zeroed = zero(T)
    for index in forcing_indices(arch)
        foreachparameters(x -> fill!(x, zero(eltype(x))), values(params(nn))[index])
        zeroed += foldparameters((n, x) -> n + sum(abs, x), zero(T), values(params(nn))[index])
    end
    @test zeroed == 0

    gradient = [i for i in eachindex(layers) if !(layers[i] isa ForcingLayer)]
    leaves = map(i -> values(params(nn))[i], gradient)
    kept_parameters = NetworkParameters(
        NamedTuple{Tuple(Symbol("L$i") for i in eachindex(leaves))}(Tuple(leaves)))
    gradient_chain = Chain(map(i -> layers[i], gradient)...)

    tolerance = 10 * eps(T)
    z = (q = rand(T, HALF), p = rand(T, HALF))
    output = nn(z)
    expected = gradient_chain(z, kept_parameters)
    @test isapprox(vcat(output.q, output.p), vcat(expected.q, expected.p); rtol = tolerance)

    x = rand(T, DIM)
    @test isapprox(nn(x), gradient_chain(x, kept_parameters); rtol = tolerance)
end

@testset "train! trains a ForcedSympNet on a DataLoader ($T)" for T in REAL_ELTYPES
    Random.seed!(6)
    arch = ForcedSympNet(DIM; n_layers = 4, upscaling_dimension = UPSCALING, forcing_type = :P)
    nn = NeuralNetwork(arch, CPU(), T)
    dl = DataLoader(rand(T, DIM, 12, 3); suppress_info = true)
    opt = TrainingOptimizer(params(nn); algorithm = Adam())

    # one epoch moves the parameters of a forcing layer, which only happens when the gradient
    # reaches the `Chain` the layer holds
    forcing_parameters = values(params(nn))[forcing_indices(arch)[1]]
    before = deepcopy(forcing_parameters)
    train!(nn, opt, dl, Batch(4), 1; show_progress = false)
    @test forcing_parameters != before

    history = train!(nn, opt, dl, Batch(4), 2; show_progress = false)
    @test length(history) == 2
    @test eltype(history) === T
    @test all(isfinite, history)
end

@testset "a forced SympNet refuses an unknown forcing type and an odd dimension" begin
    @test_throws ArgumentError ForcedSympNet(DIM; forcing_type = :X)
    @test_throws ArgumentError ForcedSympNet(3)
end

@testset "a DataLoader gives the forced SympNet of its input dimension ($T)" for T in REAL_ELTYPES
    dl = DataLoader(rand(T, DIM, 10); suppress_info = true)
    @test ForcedSympNet(dl; n_layers = 3, forcing_type = :Q, upscaling_dimension = UPSCALING) ===
          ForcedSympNet(DIM; n_layers = 3, forcing_type = :Q, upscaling_dimension = UPSCALING)
end

@testset "the forced names are public and not exported" begin
    for name in (:ForcedSympNet, :ForcingLayer, :ForcingLayerQ, :ForcingLayerP, :ForcingLayerQP)
        @test Base.ispublic(GeometricMachineLearning, name)
        @test !Base.isexported(GeometricMachineLearning, name)
    end
end
