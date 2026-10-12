using GeometricMachineLearning
using GeometricMachineLearning: params
using Test
import Random

include("../helpers/eltypes.jl")

# Each architecture that the manual trains, with the data shape and the batch its tutorial uses. The
# list holds a `SymmetricMatrix` leaf (`LASympNet`, `LinearSymplecticTransformer`), the strictly
# triangular leaves (`VolumePreservingFeedForward`, `VolumePreservingTransformer`), a Stiefel leaf
# (`SymplecticAutoencoder`), and the three `Chain`s of the volume-preserving-attention tutorial.
# The last place holds the loss; `nothing` trains on the default loss.
function manual_architectures(T)
    feedforward = DataLoader(rand(T, 4, 12, 3); suppress_info = true)
    transformer_batch = Batch(4, 3)
    hnn = StandardHamiltonianArchitecture(2, 4)
    attention_data = DataLoader(rand(T, 1, 12, 2); suppress_info = true)
    attention_batch = Batch(4, 3, 1)
    attention_chain(skew_sym) = Chain(Dense(1, 2, tanh),
        VolumePreservingAttention(2, 3; skew_sym), Dense(2, 1, identity; use_bias = true))
    (("GSympNet", GSympNet(4; n_layers = 2), feedforward, Batch(4), nothing),
        ("LASympNet", LASympNet(4; depth = 1, nhidden = 1), feedforward, Batch(4), nothing),
        ("ResNet", ResNet(4, 2, tanh), feedforward, Batch(4), nothing),
        ("VolumePreservingFeedForward",
            VolumePreservingFeedForward(4, 2), feedforward, Batch(4),
            nothing),
        ("StandardTransformerIntegrator",
            StandardTransformerIntegrator(4; n_heads = 2, L = 1, n_blocks = 1),
            feedforward, transformer_batch, nothing),
        ("LinearSymplecticTransformer",
            LinearSymplecticTransformer(4, 3; n_sympnet = 1, L = 1),
            feedforward, transformer_batch, nothing),
        ("SymplecticTransformer", SymplecticTransformer(4; n_sympnet = 1, L = 1),
            feedforward, transformer_batch, nothing),
        ("VolumePreservingTransformer", VolumePreservingTransformer(4, 3; n_blocks = 1),
            feedforward, transformer_batch, nothing),
        ("SymplecticAutoencoder",
            SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2),
            DataLoader(rand(T, 4, 12); autoencoder = true, suppress_info = true), Batch(4),
            nothing),
        ("StandardHamiltonianArchitecture", hnn,
            DataLoader(rand(T, 2, 12), rand(T, 2, 12); suppress_info = true), Batch(4),
            HNNLoss(hnn)),
        ("VolumePreservingAttention, skew_sym = true", attention_chain(true),
            attention_data, attention_batch, TransformerLoss(attention_batch)),
        ("VolumePreservingAttention, skew_sym = false", attention_chain(false),
            attention_data, attention_batch, TransformerLoss(attention_batch)),
        ("Dense comparison of the attention tutorial",
            Chain(Dense(1, 2, tanh), Dense(2, 1, identity; use_bias = true)),
            attention_data, Batch(4), FeedForwardLoss()))
end

@testset "train! trains every architecture of the manual ($T)" for T in REAL_ELTYPES
    Random.seed!(6)
    for (name, arch, dl, batch, loss) in manual_architectures(T)
        @testset "$name" begin
            nn = NeuralNetwork(arch, T)
            p₀ = deepcopy(params(nn))
            opt = TrainingOptimizer(nn.params; algorithm = Adam())
            history = loss === nothing ?
                      train!(nn, opt, dl, batch, 2; show_progress = false) :
                      train!(nn, opt, dl, batch, 2; loss, show_progress = false)
            @test length(history) == 2
            @test eltype(history) === T
            @test all(isfinite, history)
            @test params(nn) != p₀
        end
    end
end
