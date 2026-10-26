# The forcing layers add a learned term to `p` and leave `q` alone; the `Q`/`P`/`QP` of the type
# says what the term reads. The expected forcing is the layer's own `Chain` evaluated on that state
# with the flattened system parameters below it, which is what the layer is a wrapper around; the
# last layer of that chain has `dim ÷ 2` rows, which is what keeps `q` untouched.

using GeometricMachineLearning
using GeometricMachineLearning: ForcingLayer, ForcingLayerQ, ForcingLayerP, ForcingLayerQP,
                                NullParameters, concatenate_array_with_parameters, params
using AbstractNeuralNetworks: NamedTupleOfArrays
using NeuralNetworkParameters: foldparameters
using Test
import Random

include("../helpers/eltypes.jl")

const DIM = 4
const HALF = DIM ÷ 2
const SYSTEM_PARAMETERS = (m = 1.0, ω = π / 2)

# the three forcing layers, with the symbol of the type and the name each one has in a label
const FORCING_LAYERS = (("Q", :Q, ForcingLayerQ), ("P", :P, ForcingLayerP),
    ("QP", :QP, ForcingLayerQP))

# the flat vector of the system parameters, written out by hand
flat(::Type{T}) where {T} = T[SYSTEM_PARAMETERS.m, SYSTEM_PARAMETERS.ω]

# the state a forcing layer of this type reads, as the type says it
read_state(::Val{:Q}, z) = z.q
read_state(::Val{:P}, z) = z.p
read_state(::Val{:QP}, z) = vcat(z.q, z.p)

# the components of the state a forcing layer of this type reads
dependent(::Val{:Q}) = (:q,)
dependent(::Val{:P}) = (:p,)
dependent(::Val{:QP}) = (:q, :p)

# the layer of a forcing type, built through its public constructor
forcing_layer(constructor; kwargs...) = constructor(DIM, 8, 1, tanh; kwargs...)

# the parameters of a layer, as `Chain(layer)` initialises them
function layer_parameters(layer, ::Type{T}) where {T}
    params(NeuralNetwork(Chain(layer), CPU(), T)).L1
end

@testset "ForcingLayer$name returns q and p plus the forcing ($T)" for (name, FT, constructor) in FORCING_LAYERS,
    T in REAL_ELTYPES

    Random.seed!(1)
    q, p = rand(T, HALF), rand(T, HALF)
    z = (q = q, p = p)

    # without system parameters
    layer = forcing_layer(constructor)
    ps = layer_parameters(layer, T)
    expected = layer.model(read_state(Val(FT), z), ps)
    for output in (layer(z, ps), layer((z, NullParameters()), ps))
        @test output.q == z.q
        @test size(output.p) == size(z.p)
        @test eltype(output.p) === T
        @test isapprox(output.p, z.p + expected; rtol = 10 * eps(T))
        @test isapprox(output.p - z.p, expected; rtol = 10 * eps(T))
    end
    @test isapprox(layer(vcat(q, p), ps), vcat(layer(z, ps).q, layer(z, ps).p))
    @test_throws DimensionMismatch layer(vcat(q, p, q), ps)

    # with system parameters, which the layer takes beside the state it reads
    layer = forcing_layer(constructor; parameters = SYSTEM_PARAMETERS)
    ps = layer_parameters(layer, T)
    expected = layer.model(
        concatenate_array_with_parameters(read_state(Val(FT), z), SYSTEM_PARAMETERS), ps)
    @test isapprox(layer.model(vcat(read_state(Val(FT), z), flat(T)), ps), expected;
        rtol = 10 * eps(T))
    output = layer((z, SYSTEM_PARAMETERS), ps)
    @test output.q == z.q
    @test eltype(output.p) === T
    @test isapprox(output.p - z.p, expected; rtol = 10 * eps(T))
    @test isapprox(layer((vcat(q, p), SYSTEM_PARAMETERS), ps), vcat(output.q, output.p))
end

# A matrix input is a batch of states that share one set of system parameters: the parameters are
# repeated below every column, and the forcing has one column per state.
@testset "ForcingLayer$name on a matrix of states ($T)" for (name, FT, constructor) in FORCING_LAYERS,
    T in REAL_ELTYPES

    Random.seed!(2)
    z = (q = rand(T, HALF, 3), p = rand(T, HALF, 3))
    layer = forcing_layer(constructor; parameters = SYSTEM_PARAMETERS)
    ps = layer_parameters(layer, T)
    expected = layer.model(
        concatenate_array_with_parameters(read_state(Val(FT), z), SYSTEM_PARAMETERS), ps)
    output = layer((z, SYSTEM_PARAMETERS), ps)
    @test size(output.p) == size(z.p)
    @test eltype(output.p) === T
    @test isapprox(output.p - z.p, expected; rtol = 10 * eps(T))
    @test isapprox(layer((vcat(z.q, z.p), SYSTEM_PARAMETERS), ps), vcat(output.q, output.p))

    # without system parameters the input is the state the forcing reads as it is, and the layer
    # takes a matrix either bare or beside `NullParameters()`
    layer = forcing_layer(constructor)
    ps = layer_parameters(layer, T)
    expected = layer.model(read_state(Val(FT), z), ps)
    for output in (layer(z, ps), layer((z, NullParameters()), ps))
        @test output.q == z.q
        @test size(output.p) == size(z.p)
        @test eltype(output.p) === T
        @test isapprox(output.p - z.p, expected; rtol = 10 * eps(T))
        @test isapprox(layer(vcat(z.q, z.p), ps), vcat(output.q, output.p))
    end
end

@testset "ForcingLayer$name returns μ with return_parameters ($T)" for (name, FT, constructor) in FORCING_LAYERS,
    T in REAL_ELTYPES

    Random.seed!(3)
    z = (q = rand(T, HALF), p = rand(T, HALF))
    layer = forcing_layer(constructor; parameters = SYSTEM_PARAMETERS,
        return_parameters = true)
    ps = layer_parameters(layer, T)
    output, μ = layer((z, SYSTEM_PARAMETERS), ps)
    @test μ === SYSTEM_PARAMETERS
    @test output.q == z.q
    @test eltype(output.p) === T

    output, μ = layer((vcat(z.q, z.p), SYSTEM_PARAMETERS), ps)
    @test μ === SYSTEM_PARAMETERS
    @test size(output) == (DIM,)
    @test eltype(output) === T

    # a matrix of states returns `μ` beside a matrix of outputs, in both the named-tuple and the
    # stacked form; the expected forcing repeats the flat system parameters below every column
    z = (q = rand(T, HALF, 3), p = rand(T, HALF, 3))
    expected = layer.model(
        vcat(read_state(Val(FT), z), repeat(flat(T), 1, size(z.q, 2))), ps)
    output, μ = layer((z, SYSTEM_PARAMETERS), ps)
    @test μ === SYSTEM_PARAMETERS
    @test size(output.q) == size(z.q)
    @test size(output.p) == size(z.p)
    @test eltype(output.p) === T
    @test isapprox(output.p - z.p, expected; rtol = 10 * eps(T))

    output, μ = layer((vcat(z.q, z.p), SYSTEM_PARAMETERS), ps)
    @test μ === SYSTEM_PARAMETERS
    @test size(output) == (DIM, 3)
    @test eltype(output) === T
end

# A forcing layer holds its network in a type parameter, so the forward pass infers the type of its
# output on every form it takes; a layer whose `model` is typed `Chain` leaves it `Any`.
@testset "the forward pass of ForcingLayer$name infers ($T)" for (name, _, constructor) in FORCING_LAYERS,
    T in REAL_ELTYPES

    Random.seed!(4)
    q, p = rand(T, HALF), rand(T, HALF)
    z = (q = q, p = p)
    z_matrix = (q = rand(T, HALF, 3), p = rand(T, HALF, 3))

    # without system parameters: a stacked vector, a stacked matrix, and a state beside
    # `NullParameters()`
    layer = forcing_layer(constructor)
    ps = layer_parameters(layer, T)
    @test eltype(@inferred layer(vcat(q, p), ps)) === T
    @test eltype(@inferred layer(vcat(z_matrix.q, z_matrix.p), ps)) === T
    output = @inferred layer((z, NullParameters()), ps)
    @test eltype(output.p) === T

    # with system parameters: a state beside them, and a stacked state beside them
    layer = forcing_layer(constructor; parameters = SYSTEM_PARAMETERS)
    ps = layer_parameters(layer, T)
    output = @inferred layer((z, SYSTEM_PARAMETERS), ps)
    @test eltype(output.p) === T
    @test eltype(@inferred layer((vcat(q, p), SYSTEM_PARAMETERS), ps)) === T

    # the form that returns the system parameters beside the output
    layer = forcing_layer(constructor; parameters = SYSTEM_PARAMETERS,
        return_parameters = true)
    ps = layer_parameters(layer, T)
    output, μ = @inferred layer((z, SYSTEM_PARAMETERS), ps)
    @test μ === SYSTEM_PARAMETERS
    @test eltype(output.p) === T
end

# The suffix names what the forcing *reads*: a force enters the `ṗ` equation, so all three types add
# to `p`, and a perturbation of a component the type does not read leaves the forcing as it was.
@testset "ForcingLayer$name reads only the component of its type ($T)" for (name, FT, constructor) in FORCING_LAYERS,
    T in REAL_ELTYPES

    Random.seed!(1234)
    layer = forcing_layer(constructor)
    ps = layer_parameters(layer, T)
    z = (q = rand(T, HALF), p = rand(T, HALF))
    forcing = layer(z, ps).p - z.p

    for coordinate in (:q, :p)
        perturbed = merge(z, NamedTuple{(coordinate,)}((z[coordinate] .+ 1,)))
        moved = !isapprox(layer(perturbed, ps).p - perturbed.p, forcing; rtol = 10 * eps(T))
        @test moved == (coordinate in dependent(Val(FT)))
    end
    @test layer(z, ps).q == z.q
end

# The entries `initialparameters` allocates are the input layer, the hidden layers and the linear map
# back to `dim ÷ 2` rows; the closed form beside the counted one is written out from those three.
@testset "parameterlength of a forcing layer is what it allocates ($T)" for T in REAL_ELTYPES
    Random.seed!(8)
    for layer in (ForcingLayerQ(DIM), ForcingLayerP(DIM, 8, 2),
        ForcingLayerQP(DIM, 8, 1, tanh; parameters = SYSTEM_PARAMETERS))
        ps = layer_parameters(layer, T)
        @test parameterlength(layer) == foldparameters((n, x) -> n + length(x), 0, ps)
    end

    # the input layer over the read state and the two system parameters, `nhidden` hidden layers of
    # `width`, and the linear map back to `dim ÷ 2` rows
    for (FT, input_width) in ((:Q, HALF), (:P, HALF), (:QP, DIM))
        layer = ForcingLayer(DIM, 8, 2, tanh; parameters = SYSTEM_PARAMETERS, type = FT)
        @test parameterlength(layer) ==
              (input_width + 2) * 8 + 8 + 2 * (8 * 8 + 8) + 8 * HALF
    end
end

@testset "a forcing layer refuses an odd dimension and an unknown forcing type" begin
    Random.seed!(5)
    @test_throws ArgumentError ForcingLayerQ(3)
    @test_throws ArgumentError ForcingLayerP(3)
    @test_throws ArgumentError ForcingLayerQP(3)
    @test_throws ArgumentError ForcingLayer(4, 4, 1, tanh; type = :X)

    # the layer reads `dim ÷ 2` rows of each half, so a dimension of the wrong size is a
    # `DimensionMismatch` at the call
    layer = ForcingLayerQ(DIM)
    ps = layer_parameters(layer, Float64)
    @test_throws DimensionMismatch layer((q = rand(HALF + 1), p = rand(HALF)), ps)
end

# A tuple that is neither a state, nor a state with its system parameters, matches no method:
# there is no catch-all.
@testset "a forcing layer has no catch-all" begin
    Random.seed!(6)
    layer = ForcingLayerQ(DIM)
    ps = layer_parameters(layer, Float64)
    @test_throws MethodError layer((rand(HALF), rand(HALF)), ps)
    @test_throws MethodError layer((q = rand(HALF),), ps)
end
