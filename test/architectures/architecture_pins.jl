# The forward pass of each architecture whose chain is built from shared code, and the outputs of
# the encoders and decoders, against the outputs that commit 4c4a8320 records for the same seed and
# input; and the layer types of `LASympNet` for its four `(init_upper_linear, init_upper_act)`
# pairs, and the encoder and decoder types of the three autoencoder families.

using GeometricMachineLearning
using GeometricMachineLearning: AutoEncoder, NonLinearSymplecticEncoder,
                                NonLinearSymplecticDecoder,
                                UnknownSymplecticEncoder, UnknownSymplecticDecoder,
                                UnknownEncoder, UnknownDecoder
using Test

include("../helpers/eltypes.jl")
include("../helpers/pins.jl")
include("../helpers/pinned_outputs.jl")

@testset "$name ($T)" for (i, (
        name, model, dim, cols)) in enumerate(architecture_pin_cases()),
    T in REAL_ELTYPES

    nn = pinned_network(model, T, architecture_seed(i))
    y = nn(pin_input(T, dim, cols))
    @test eltype(y) === T
    @test within_ulps(vec(y), ARCHITECTURE_PINS[(name, T)])
end

@testset "encoder and decoder of $name ($T)" for (i, (
        name, model, full, reduced)) in enumerate(autoencoder_pin_cases()),
    T in REAL_ELTYPES

    nn = pinned_network(model, T, autoencoder_seed(i))
    ye = encoder(nn)(pin_input(T, full, 3))
    yd = decoder(nn)(pin_input(T, reduced, 3))
    @test eltype(ye) === T
    @test eltype(yd) === T
    @test within_ulps(vec(ye), AUTOENCODER_PINS[(name, :encoder, T)])
    @test within_ulps(vec(yd), AUTOENCODER_PINS[(name, :decoder, T)])
end

@testset "LASympNet layer types, init_upper_linear = $linear, init_upper_act = $act" for linear in (true,
        false),
    act in (true, false)

    arch = LASympNet(
        4; depth = 3, nhidden = 2, init_upper_linear = linear, init_upper_act = act)
    @test [type_string(typeof(l)) for l in Chain(arch).layers] ==
          LASYMPNET_LAYER_TYPES[(linear, act)]
end

# An autoencoder that defines its layers and nothing else, so that `encoder` and `decoder` reach the
# methods for a plain `AutoEncoder`.
struct PlainAE <: AutoEncoder
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
    n_decoder_blocks::Int
end

function GeometricMachineLearning.encoder_layers_from_iteration(arch::PlainAE, ::AbstractVector{<:Integer})
    (
        Dense(arch.full_dim, arch.reduced_dim, tanh),)
end
function GeometricMachineLearning.decoder_layers_from_iteration(arch::PlainAE, ::AbstractVector{<:Integer})
    (
        Dense(arch.reduced_dim, arch.full_dim, tanh),)
end

@testset "encoder and decoder types of each autoencoder family ($T)" for T in REAL_ELTYPES
    sae = NeuralNetwork(SymplecticAutoencoder(8, 4), CPU(), T)
    @test encoder(sae).architecture isa NonLinearSymplecticEncoder
    @test decoder(sae).architecture isa NonLinearSymplecticDecoder

    psd = NeuralNetwork(PSDArch(8, 4), CPU(), T)
    @test encoder(psd).architecture isa UnknownSymplecticEncoder
    @test decoder(psd).architecture isa UnknownSymplecticDecoder

    plain = NeuralNetwork(PlainAE(6, 2, 2, 2), CPU(), T)
    @test encoder(plain).architecture isa UnknownEncoder
    @test decoder(plain).architecture isa UnknownDecoder
    @test eltype(decoder(plain)(encoder(plain)(pin_input(T, 6, 3)))) === T
end
