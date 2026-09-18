using Test
using GeometricMachineLearning
using GeometricMachineLearning: initialparameters
using AbstractNeuralNetworks: parameterlength
using KernelAbstractions
import Random
import Zygote

Random.seed!(1234)

@testset "the encoding matrix is the formula of the paper" begin
    for (dim, seq_length) in ((4, 6), (8, 3), (2, 10))
        P = positional_encoding(Float64, dim, seq_length)
        @test size(P) == (dim, seq_length)

        # Recomputed here from the paper rather than from the implementation: an assertion that
        # rebuilds the code under test proves only that it is deterministic.
        for i in 1:dim, pos in 1:seq_length

            θ = (pos - 1) / 10000.0^(2 * ((i - 1) ÷ 2) / dim)
            @test P[i, pos] ≈ (isodd(i) ? sin(θ) : cos(θ))
        end
    end
end

@testset "positions are counted from zero, and row pairs share a frequency" begin
    P = positional_encoding(Float64, 6, 4)

    # The first position is zero, so its column alternates sin(0), cos(0).
    @test P[:, 1] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    # Rows 2j+1 and 2j+2 are the sine and cosine of the same angle, so their squares sum to one.
    # This is what "a pair of rows encodes the position at one wavelength" means, and it is what
    # discriminates the correct pairing from one that is off by one.
    for j in 0:2, pos in 1:4

        @test P[2 * j + 1, pos]^2 + P[2 * j + 2, pos]^2 ≈ 1.0
    end
end

@testset "the element type is honoured" begin
    @test eltype(positional_encoding(Float32, 4, 3)) == Float32
    @test eltype(positional_encoding(Float64, 4, 3)) == Float64
    @test eltype(PositionalEncoding(4)(zeros(Float32, 4, 3), NamedTuple())) == Float32
end

@testset "the layer adds the encoding and has no parameters" begin
    l = PositionalEncoding(4)
    x = rand(Float32, 4, 5)

    # Against a literal rather than against `x .+ positional_encoding(...)`, which would restate the
    # functor's body — the thing the first testset's comment warns about. Column 1 of the
    # encoding is `(0, 1, 0, 1)`, so the layer leaves the odd rows of the first column alone and
    # adds one to the even rows.
    @test l(x, NamedTuple())[:, 1] ≈ x[:, 1] .+ Float32[0, 1, 0, 1]
    @test parameterlength(l) == 0
    @test initialparameters(Random.default_rng(),
        GeometricMachineLearning.AbstractNeuralNetworks.DefaultInitializer(),
        l, CPU(), Float32) == NamedTuple()
end

@testset "the sequence length comes from the input, not the constructor" begin
    l = PositionalEncoding(4)
    # The same layer applied to two different lengths, which is the reason it stores no length: a
    # network here is applied to trajectories of whatever length the data has.
    @test size(l(rand(Float32, 4, 3), NamedTuple())) == (4, 3)
    @test size(l(rand(Float32, 4, 9), NamedTuple())) == (4, 9)

    # And over a batch, broadcasting along the third axis.
    x = rand(Float32, 4, 5, 2)
    y = l(x, NamedTuple())
    @test size(y) == (4, 5, 2)
    @test y[:, :, 1] ≈ x[:, :, 1] .+ positional_encoding(Float32, 4, 5)
    @test y[:, :, 2] ≈ x[:, :, 2] .+ positional_encoding(Float32, 4, 5)
end

@testset "the Jacobian is the identity" begin
    # This is the claim that lets the layer sit in front of a structure-preserving architecture: it
    # adds a constant, so it changes no derivative, so it preserves whatever the layers after it
    # preserve.
    #
    # Not asserted with `==`: the identity `(x + δ + P) - (x + P) = δ` is exact in ℝ and not in
    # floating point, because each addition rounds and the two round differently. The deviation is
    # at the last bit, which is what the bound below says.
    l = PositionalEncoding(6)
    x = rand(Float64, 6, 4)
    δ = rand(Float64, 6, 4)
    difference = l(x + δ, NamedTuple()) - l(x, NamedTuple())
    @test maximum(abs, difference - δ) < 8 * eps(Float64)
end

@testset "a (q, p) input is accepted, and agrees with the stacked one" begin
    # The keyword must not *remove* an input type the transformer otherwise takes.
    # `MultiHeadAttention` and `ResNetLayer` both stack such an input, so this layer does too, and
    # the two paths have to give the same answer.
    l = PositionalEncoding(4)
    z = (q = rand(Float32, 2, 5), p = rand(Float32, 2, 5))

    split = l(z, NamedTuple())
    @test split isa NamedTuple{(:q, :p)}
    @test size(split.q) == (2, 5)
    @test vcat(split.q, split.p) == l(vcat(z.q, z.p), NamedTuple())

    for kw in (false, true)
        nn = NeuralNetwork(Transformer(4, 2, 1; positional_encoding = kw), CPU(), Float32)
        @test nn(z) isa NamedTuple{(:q, :p)}
    end
end

@testset "the keyword does not change which inputs the transformer takes" begin
    # A vector is rejected with the keyword exactly as it is without it. Broadcasting a vector
    # against the `4 × 1` encoding would return a `4 × 1` matrix, so accepting one here would let
    # the keyword change the rank of the output.
    l = PositionalEncoding(4)
    @test_throws MethodError l(rand(Float32, 4), NamedTuple())

    for kw in (false, true)
        nn = NeuralNetwork(Transformer(4, 2, 1; positional_encoding = kw), CPU(), Float32)
        @test_throws MethodError nn(rand(Float32, 4))
    end
end

@testset "a network carrying the layer can be trained" begin
    # The encoding is a constant, built with a mutating loop. Zygote refuses to differentiate that
    # loop unless the builder is declared non-differentiable, and without the declaration every
    # gradient through a network with `positional_encoding = true` raised
    # `Mutating arrays is not supported`. A layer that cannot be trained through is not a layer.
    x = rand(Float32, 4, 5)

    for kw in (false, true)
        nn = NeuralNetwork(Transformer(4, 2, 1; positional_encoding = kw), CPU(), Float32)

        # Both gradients are taken, because the encoding sits between the input and the parameters
        # and the failure was in the forward trace, which both of them walk.
        @test Zygote.gradient(p -> sum(nn(x, p)), nn.params)[1] !== nothing

        gradient_wrt_input = Zygote.gradient(y -> sum(nn(y, nn.params)), x)[1]
        @test size(gradient_wrt_input) == size(x)
        @test all(isfinite, gradient_wrt_input)
    end
end

@testset "the Transformer keyword puts the layer at the front, and only then" begin
    with_encoding = Transformer(4, 2, 2; positional_encoding = true)
    without = Transformer(4, 2, 2)

    @test with_encoding.layers[1] isa PositionalEncoding
    @test !any(l -> l isa PositionalEncoding, without.layers)
    @test length(with_encoding.layers) == length(without.layers) + 1

    # It survives the network constructor, whose parameter set has to cope with a layer that
    # contributes nothing to it.
    nn = NeuralNetwork(with_encoding, CPU(), Float32)
    @test nn.params[1] == NamedTuple()
    @test size(nn(rand(Float32, 4, 5))) == (4, 5)
end
