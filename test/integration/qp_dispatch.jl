# AbstractNeuralNetworks' `NamedTupleOfArrays{T, S}` ties the element types of the arrays of a named
# tuple together but not their array types, so in `(q, p)` data `q` may be a view where `p` is an
# `Array`. The layers, the losses, `ZygotePullback`, `_diff` and the `DataLoader` take such data,
# and the expected value of each call is the same call on two `Array`s. The methods that read the
# keys `q` and `p` refuse a named tuple with other keys; `map_to_type`, `map_to_new_backend` and
# `networkbackend`, which only map over a named tuple or read its first array, take any keys.

using GeometricMachineLearning
using GeometricMachineLearning: ResNetLayer, _diff, ZygotePullback
using ReducedComplexityModeling: map_to_type, map_to_new_backend
using AbstractNeuralNetworks: networkbackend, params
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

@testset "(q, p) methods take q and p of different array types ($T)" for T in REAL_ELTYPES
    Random.seed!(11)
    # `q` is a view into a larger tensor, `p` an `Array`; `qp` holds the same numbers in two `Array`s
    q = view(rand(T, 2, 3, 3), :, :, 1:2)
    p = rand(T, 2, 3, 2)
    mixed = (q = q, p = p)
    qp = (q = collect(q), p = p)
    @test typeof(mixed.q) !== typeof(mixed.p)

    dense = NeuralNetwork(Chain(Dense(4, 4, tanh)), T)
    resnet = NeuralNetwork(Chain(ResNetLayer(4, tanh)), T)
    attention = NeuralNetwork(Chain(MultiHeadAttention(4, 2)), T)
    encoding = NeuralNetwork(Chain(PositionalEncoding(4)), T)
    for nn in (dense, resnet, attention, encoding)
        @test nn(mixed) == nn(qp)
        @test eltype(nn(mixed).q) === T
    end

    loss = TransformerLoss(3, 3)
    @test loss(attention.model, params(attention), mixed, mixed) ==
          loss(attention.model, params(attention), qp, qp)
    # the input and the output need not have one type either
    @test loss(attention.model, params(attention), mixed, qp) ==
          loss(attention.model, params(attention), qp, qp)

    autoencoder_loss = AutoEncoderLoss()
    @test autoencoder_loss(dense.model, params(dense), mixed) ==
          autoencoder_loss(dense.model, params(dense), qp)
    @test autoencoder_loss(dense, mixed) == autoencoder_loss(dense, qp)
    # the four-argument method reconstructs its input and ignores the output
    @test autoencoder_loss(dense.model, params(dense), mixed, qp) ==
          autoencoder_loss(dense.model, params(dense), qp)
    reduced_loss = ReducedLoss(encoder(NeuralNetwork(Chain(Dense(4, 4, tanh)), T)),
        decoder(NeuralNetwork(Chain(Dense(4, 4, tanh)), T)))
    @test reduced_loss(dense.model, params(dense), mixed, mixed) ==
          reduced_loss(dense.model, params(dense), qp, qp)
    @test reduced_loss(dense.model, params(dense), mixed, qp) ==
          reduced_loss(dense.model, params(dense), qp, qp)

    # both shapes of `ZygotePullback`: an input alone, and an (input, output) pair
    for (pullback, data_mixed, data_qp) in (
        (ZygotePullback(AutoEncoderLoss()), mixed, qp),
        (ZygotePullback(FeedForwardLoss()), (mixed, mixed), (qp, qp)))
        value_mixed, back_mixed = pullback(params(dense), dense.model, data_mixed)
        value_qp, back_qp = pullback(params(dense), dense.model, data_qp)
        @test value_mixed == value_qp
        @test back_mixed(one(T)).L1.W ≈ back_qp(one(T)).L1.W
    end

    𝕁 = poisson_matrix(4, T)
    z = (q = view(T[1, 2, 0], 1:2), p = T[3, 4])
    @test 𝕁 * vcat(z.q, z.p) == vcat(z.p, -z.q)
    @test _diff(mixed, mixed) == (q = zero(qp.q), p = zero(qp.p))

    # a loader on `mixed` converts to the other precision: `networkbackend`, `map_to_type` and
    # `map_to_new_backend` all take it
    dl = DataLoader{T, typeof(mixed), Nothing, :TimeSeries}(
        mixed, nothing, 4, 3, 2, nothing, nothing)
    @test networkbackend(dl) == CPU()
    S = T === Float32 ? Float64 : Float32
    converted = DataLoader(dl, S)
    @test eltype(converted) === S
    @test converted.input == (q = S.(qp.q), p = S.(qp.p))
end

@testset "(q, p) methods refuse a named tuple with other keys" begin
    ab = (a = rand(2, 3, 2), b = rand(2, 3, 2))
    for layer in (Dense(4, 4, tanh), ResNetLayer(4, tanh), MultiHeadAttention(4, 2),
        PositionalEncoding(4))
        nn = NeuralNetwork(Chain(layer))
        @test_throws MethodError nn(ab)
    end
    # `TransformerLoss` has no method for `ab`, so AbstractNeuralNetworks' fallback for a loss raises.
    # The message is the assertion: `@test_throws ErrorException` also passes on a `FieldError`.
    attention = NeuralNetwork(Chain(MultiHeadAttention(4, 2)))
    @test_throws "Functor not defined for `NetworkLoss`" TransformerLoss(3, 3)(
        attention.model, params(attention), ab, ab)
    @test_throws MethodError poisson_matrix(4) * (a = [1.0, 2.0], b = [3.0, 4.0])
    # the generic `_diff` of two named tuples differences each entry
    @test _diff(ab, ab) == (a = zero(ab.a), b = zero(ab.b))
end

@testset "map_to_type, map_to_new_backend and networkbackend take any named tuple of arrays ($T)" for T in REAL_ELTYPES
    Random.seed!(12)
    ab = (a = rand(2, 3), b = rand(2, 3))
    converted = map_to_type(ab, T)
    @test keys(converted) == (:a, :b)
    @test converted.a == T.(ab.a)
    @test eltype(converted.b) === T
    @test map_to_new_backend(ab, CPU()) === ab

    # the copying constructor's default backend is the backend of the loader's first array
    ab3 = (a = rand(2, 3, 2), b = rand(2, 3, 2))
    dl = DataLoader{Float64, typeof(ab3), Nothing, :TimeSeries}(
        ab3, nothing, 4, 3, 2, nothing, nothing)
    @test networkbackend(dl) == CPU()
    converted_dl = DataLoader(dl, T)
    @test eltype(converted_dl) === T
    @test converted_dl.input == (a = T.(ab3.a), b = T.(ab3.b))
end
