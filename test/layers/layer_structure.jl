# The volume-preserving, symplectic-attention and ResNet layers and the symplectic autoencoder
# encode each structural choice as a `Symbol` type parameter, chosen in one expression from the
# public `Bool` keyword; a `Q`/`P` pair, a lower/upper pair and the two weight kinds of
# multi-head attention share one method; the dimension field of a gradient layer is concrete; and
# every type argument under `src/layers/` is spelled `::Type{T}`.

using GeometricMachineLearning
using GeometricMachineLearning: GradientLayer, LinearLayer, ResNetLayer, initialparameters,
                                params
using AbstractNeuralNetworks: parameterlength, ZeroInitializer, OneInitializer
using Test
import Random

const GML = GeometricMachineLearning

include("../helpers/eltypes.jl")

# The methods of `f` whose definitions are in the file `file` under `src/`.
function methods_in(f, file::AbstractString)
    filter(
        m -> endswith(String(m.file), joinpath("src", file)), collect(methods(f)))
end

# The forward methods on a `(q, p)` input that the layers `layers` reach in the file `file`: the
# methods whose input argument is not an array.
function qp_forward_methods(layers, file::AbstractString)
    found = Set{Method}()
    for l in layers, m in methods_in(l, file)

        input = Base.unwrap_unionall(m.sig).parameters[2]
        input <: AbstractArray || push!(found, m)
    end
    found
end

@testset "a Q/P pair shares its forward method" begin
    sympnet_layers = (GradientLayerQ(4, 6, tanh), GradientLayerP(4, 6, tanh),
        ActivationLayerQ(4, tanh), ActivationLayerP(4, tanh), LinearLayerQ(4), LinearLayerP(4))
    @test length(qp_forward_methods(sympnet_layers, joinpath("layers", "sympnets.jl"))) == 3

    attention_layers = (SymplecticAttentionQ(4; symmetric = true),
        SymplecticAttentionP(4; symmetric = true), SymplecticAttentionQ(4; symmetric = false),
        SymplecticAttentionP(4; symmetric = false))
    @test length(qp_forward_methods(attention_layers,
        joinpath("layers", "symplectic_attention.jl"))) == 2

    linear_attention_layers = (
        LinearSymplecticAttentionQ(4, 3), LinearSymplecticAttentionP(4, 3))
    @test length(qp_forward_methods(linear_attention_layers,
        joinpath("layers", "linear_symplectic_attention.jl"))) == 1
end

# The number of type variables of the alias `t`. Julia drops a variable that the body of an alias
# does not use, so an unused one shows only in the source text; this pins the two that are used.
type_variables(t) = t isa UnionAll ? 1 + type_variables(t.body) : 0

@testset "the linear SympNet aliases take two parameters" begin
    @test type_variables(LinearLayerQ) == type_variables(LinearLayerP) == 2
    @test LinearLayerQ{4, 4} === LinearLayer{4, 4, :Q}
    @test LinearLayerP{4, 4} === LinearLayer{4, 4, :P}
end

@testset "a SympNet layer raises on a half of the wrong size ($T)" for T in REAL_ELTYPES
    # In a layer of dimension 2 the scale has length 1 and broadcasts against a half of any length,
    # so there a half of the wrong size can pass the update unless the layer checks the dimension.
    for (M, layers) in (
        (4,
        (GradientLayerQ(4, 6, tanh),
            GradientLayerP(4, 6, tanh), ActivationLayerQ(4, tanh),
            ActivationLayerP(4, tanh), LinearLayerQ(4), LinearLayerP(4))),
        (2, (ActivationLayerQ(2, tanh), ActivationLayerP(2, tanh))))
        for l in layers
            ps = params(NeuralNetwork(Chain(l), CPU(), T))[1]
            m = M ÷ 2
            for n in (m - 1, m + 1, m + 3)
                n == 0 && continue
                @test_throws ErrorException("Dimension mismatch.") l(
                    (q = ones(T, n, 3), p = ones(T, n, 3)), ps)
                @test_throws ErrorException("Dimension mismatch.") l(
                    (q = ones(T, n), p = ones(T, n)), ps)
                @test_throws ErrorException("Dimension mismatch.") l(
                    (q = ones(T, n, 3), p = ones(T, m, 3)), ps)
            end
            @test eltype(l((q = ones(T, m, 3), p = ones(T, m, 3)), ps).q) === T
        end
    end
end

@testset "a gradient layer stores its dimensions as Int" begin
    for l in (GradientLayerQ(4, 8, tanh), GradientLayerQ(Int32(4), 8, tanh),
        GradientLayerP(4, Int32(8), tanh))
        @test isconcretetype(fieldtype(typeof(l), :second_dim))
        @test l.second_dim === 8
    end
    @test GradientLayerQ(Int32(4), Int32(8), tanh) === GradientLayerQ(4, 8, tanh)
end

@testset "every Bool keyword becomes a Symbol type parameter" begin
    for skew_sym in (true, false)
        @test typeof(VolumePreservingAttention(4, 3; skew_sym)) ===
              VolumePreservingAttention{4, 4, skew_sym ? :skew_sym : :arbitrary, 3}
    end
    for symmetric in (true, false),
        (constructor, side) in ((SymplecticAttentionQ, :Q), (SymplecticAttentionP, :P))

        l = constructor(4; symmetric)
        @test typeof(l).parameters[3] === side
        @test typeof(l).parameters[4] === (symmetric ? :symmetric : :arbitrary)
    end
    for use_bias in (true, false)
        bias = use_bias ? :bias : :no_bias
        @test typeof(VolumePreservingLowerLayer(3; use_bias)) ===
              VolumePreservingLowerLayer{3, 3, bias, typeof(tanh)}
        @test typeof(VolumePreservingUpperLayer(3; use_bias)) ===
              VolumePreservingUpperLayer{3, 3, bias, typeof(tanh)}
        # `ResNetLayer` encodes the bias as the volume-preserving layers do.
        @test typeof(ResNetLayer(3; use_bias)).parameters[3] ===
              typeof(VolumePreservingLowerLayer(3; use_bias)).parameters[3] === bias
    end
    @test ResNetLayer(3) isa ResNetLayer{3, 3, :bias, typeof(identity)}
    for encoder_init_q in (true, false), decoder_init_q in (true, false)

        @test typeof(SymplecticAutoencoder(8, 4; encoder_init_q, decoder_init_q)) ===
              SymplecticAutoencoder{encoder_init_q ? :EncoderInitQ : :EncoderInitP,
            decoder_init_q ? :DecoderInitQ : :DecoderInitP, typeof(tanh)}
    end
end

@testset "ResNetLayer without and with a bias ($T)" for T in REAL_ELTYPES
    Random.seed!(1)
    l = ResNetLayer(3, tanh; use_bias = false)
    ps = params(NeuralNetwork(Chain(l), CPU(), T))[1]
    @test keys(ps) == (:weight,)
    @test parameterlength(l) == 9 == sum(length, values(ps))
    x = T[1 2; -1 0; 0.5 3]
    y = l(x, ps)
    @test eltype(y) === T
    @test y ≈ x + tanh.(ps.weight * x)

    l = ResNetLayer(3, tanh; use_bias = true)
    ps = params(NeuralNetwork(Chain(l), CPU(), T))[1]
    @test keys(ps) == (:weight, :bias)
    @test parameterlength(l) == 12 == sum(length, values(ps))
    ps = (weight = ps.weight, bias = T[1, -1, 2])
    y = l(x, ps)
    @test eltype(y) === T
    @test y ≈ x + tanh.(ps.weight * x .+ ps.bias)
end

@testset "the lower and the upper layer are one struct" begin
    @test typeof(VolumePreservingLowerLayer(3)).name ===
          typeof(VolumePreservingUpperLayer(3)).name
    @test VolumePreservingLowerLayer(3) isa VolumePreservingFeedForwardLayer
    @test VolumePreservingUpperLayer(3) isa VolumePreservingFeedForwardLayer
    @test length(methods_in(initialparameters,
        joinpath("layers", "volume_preserving_feedforward.jl"))) == 1
end

@testset "the bias keyword is init_bias in every layer ($T)" for T in REAL_ELTYPES
    rng = Random.default_rng()
    for l in (VolumePreservingLowerLayer(3), VolumePreservingUpperLayer(3), ResNetLayer(3))
        ps = initialparameters(
            rng, ZeroInitializer(), l, CPU(), T; init_bias = OneInitializer())
        @test ps.bias == ones(T, 3)
        @test eltype(ps.bias) === T
    end
    weight = initialparameters(
        rng, ZeroInitializer(), VolumePreservingLowerLayer(3), CPU(),
        T).weight
    @test weight isa StrictlyLowerTriangular{T}
    weight = initialparameters(
        rng, ZeroInitializer(), VolumePreservingUpperLayer(3), CPU(),
        T).weight
    @test weight isa StrictlyUpperTriangular{T}
end

@testset "MultiHeadAttention has one initialparameters" begin
    @test length(methods_in(initialparameters, joinpath("layers", "multi_head_attention.jl"))) ==
          1
end

@testset "MultiHeadAttention draws its weights head by head ($T)" for T in REAL_ELTYPES
    # The parameters of two heads are those of one head drawn twice in a row from the same stream:
    # for each head the query, the key and the value, in this order.
    rng = Random.Xoshiro(7)
    ps = initialparameters(rng, GML.AbstractNeuralNetworks.GlorotUniform(),
        MultiHeadAttention(4, 2), CPU(), T)
    rng = Random.Xoshiro(7)
    one_head = GML.AbstractNeuralNetworks.GlorotUniform()
    expected = [begin
                    W = zeros(T, 4, 2)
                    one_head(rng, W)
                    W
                end
                for _ in 1:6]
    @test [ps.PQ.head_1, ps.PK.head_1, ps.PV.head_1,
        ps.PQ.head_2, ps.PK.head_2, ps.PV.head_2] ==
          expected
    @test eltype(ps.PQ.head_1) === T
end

@testset "MultiHeadAttention finds each head by its name, $(ndims(x)) dimensions ($T)" for T in REAL_ELTYPES,
    x in (T[sin(i + 2j) for i in 1:6, j in 1:3], T[sin(i + 2j + 3k)
                                                   for i in 1:6, j in 1:3, k in 1:2])

    Random.seed!(3)
    d = MultiHeadAttention(6, 3)
    ps = params(NeuralNetwork(Chain(d), CPU(), T))[1]
    reordered(nt) = (head_3 = nt.head_3, head_1 = nt.head_1, head_2 = nt.head_2)
    y = d(x, ps)
    @test eltype(y) === T
    @test d(x, (PQ = reordered(ps.PQ), PK = ps.PK, PV = ps.PV)) == y
    @test d(x, (PQ = reordered(ps.PQ), PK = reordered(ps.PK), PV = reordered(ps.PV))) == y
end

@testset "every type argument in src/layers/ is spelled ::Type{T}" begin
    hits = String[]
    for file in readdir(joinpath(pkgdir(GML), "src", "layers"); join = true)
        for (n, line) in enumerate(eachline(file))
            occursin(r"T::Type\b", line) && push!(hits, "$(basename(file)):$n")
        end
    end
    @test hits == String[]
end
