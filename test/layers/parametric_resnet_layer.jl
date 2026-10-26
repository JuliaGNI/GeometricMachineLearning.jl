# `ParametricResNetLayer` appends the flattened system parameters to the input of its upscaling
# weight. The expected values here are the closed form with `vcat(x, μ)` written out by hand, so
# they do not go through `concatenate_array_with_parameters`, and the μ columns of the upscaling
# weight are deliberately nonzero: a layer that drops the parameters cannot pass.

using GeometricMachineLearning
using GeometricMachineLearning: WideResNetLayer, ParametricResNetLayer, ResNetLayer,
                                NullParameters, parameterlength
using AbstractNeuralNetworks: initialparameters, GlorotUniform
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/resnet_forward.jl")

const DIM = 4
const WIDTH = 7

# |μ| = 2 for the count of the entries.
const SYSTEM_PARAMETERS = (a = 1.0, b = 2.0)

# One `μ` of three entries and its flattened vector, written out: the first entry of a `NamedTuple`
# of parameters comes first, and an array leaf of two entries follows it.
const MU = (a = 0.5, b = [-1.5, 2.0])
const MU_VECTOR = [0.5, -1.5, 2.0]

allocated_entries(x::AbstractArray) = length(x)
allocated_entries(x::NamedTuple) = sum(allocated_entries, values(x); init = 0)

function fixed_parameters(::Type{T}, parameter_length::Int) where {T}
    rng = Random.Xoshiro(11)
    (upscale_weight = randn(rng, T, WIDTH, DIM + parameter_length),
        downscale_weight = randn(rng, T, DIM, WIDTH),
        upscale_bias = randn(rng, T, WIDTH),
        bias = randn(rng, T, DIM))
end

@testset "parameterlength(::ParametricResNetLayer) counts the entries it allocates" begin
    l = ParametricResNetLayer(DIM, WIDTH, tanh; parameters = SYSTEM_PARAMETERS)
    ps = initialparameters(Random.Xoshiro(1), GlorotUniform(), l, CPU(), Float64)
    @test l.parameter_length == 2
    @test parameterlength(l) == allocated_entries(ps)
    # the shapes are the ones the forward pass needs, so a layer drawn by `initialparameters` is
    # applicable as it is
    @test size(l(randn(Random.Xoshiro(2), DIM), SYSTEM_PARAMETERS, ps)) == (DIM,)
    @test size(l(randn(Random.Xoshiro(3), DIM, 3), SYSTEM_PARAMETERS, ps)) == (DIM, 3)
end

# A `(q, p)` pair splits the state in two, so an odd dimension cannot hold one; the parametric layer
# raises what `ResNetLayer` raises.
@testset "an odd dimension with a (q, p) pair raises as ResNetLayer does" begin
    z = (q = [1.0], p = [1.0])

    plain = ResNetLayer(3, tanh; use_bias = true)
    plain_ps = initialparameters(Random.Xoshiro(2), GlorotUniform(), plain, CPU(), Float64)
    @test_throws AssertionError plain(z, plain_ps)

    l = ParametricResNetLayer(3, 5, tanh; parameters = SYSTEM_PARAMETERS)
    ps = initialparameters(Random.Xoshiro(3), GlorotUniform(), l, CPU(), Float64)
    @test_throws AssertionError l(z, SYSTEM_PARAMETERS, ps)
end

@testset "the forward pass of ParametricResNetLayer is the closed form ($T)" for T in REAL_ELTYPES
    σ = tanh
    l = ParametricResNetLayer(DIM, WIDTH, σ; parameters = MU)
    ps = fixed_parameters(T, 3)
    μ = (a = T(MU.a), b = T.(MU.b))
    μv = T.(MU_VECTOR)

    # the μ columns stay nonzero, so the parameters cannot be dropped without changing the output
    @test !all(iszero, ps.upscale_weight[:, (DIM + 1):end])

    for input in (randn(Random.Xoshiro(2), T, DIM),
        randn(Random.Xoshiro(3), T, DIM, 3),
        randn(Random.Xoshiro(8), T, DIM, 3, 2))
        output = l(input, μ, ps)
        @test size(output) == size(input)
        @test eltype(output) === T
        @test isapprox(output, parametric_forward(input, μv, ps, σ);
            atol = 10 * eps(T), rtol = 10 * eps(T))
    end

    z = (q = randn(Random.Xoshiro(4), T, DIM ÷ 2), p = randn(Random.Xoshiro(5), T, DIM ÷ 2))
    expected = parametric_forward(vcat(z.q, z.p), μv, ps, σ)
    output = l(z, μ, ps)
    @test keys(output) == (:q, :p)
    @test isapprox(vcat(output.q, output.p), expected;
        atol = 10 * eps(T), rtol = 10 * eps(T))
end

@testset "without system parameters the layer is a WideResNetLayer ($T)" for T in REAL_ELTYPES
    σ = tanh
    ps = fixed_parameters(T, 0)
    x = randn(Random.Xoshiro(6), T, DIM, 2)
    wide = WideResNetLayer(DIM, WIDTH, σ)
    parametric = ParametricResNetLayer(DIM, WIDTH, σ)
    @test parametric.parameter_length == 0
    @test parameterlength(parametric) == parameterlength(wide)
    @test parametric(x, NullParameters(), ps) == wide(x, ps)
end

@testset "the Chain threads the system parameters through a ParametricResNetLayer ($T)" for T in REAL_ELTYPES
    σ = tanh
    l = ParametricResNetLayer(DIM, WIDTH, σ; parameters = MU, return_parameters = true)
    ps = fixed_parameters(T, 3)
    μ = (a = T(MU.a), b = T.(MU.b))
    x = randn(Random.Xoshiro(7), T, DIM)
    output, returned_μ = l((x, μ), ps)
    @test returned_μ === μ
    @test isapprox(output, parametric_forward(x, T.(MU_VECTOR), ps, σ);
        atol = 10 * eps(T), rtol = 10 * eps(T))
end

@testset "ParametricResNetLayer is public and not exported" begin
    @test Base.ispublic(GeometricMachineLearning, :ParametricResNetLayer)
    @test !Base.isexported(GeometricMachineLearning, :ParametricResNetLayer)
end
