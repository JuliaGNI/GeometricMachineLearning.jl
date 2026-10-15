using GeometricMachineLearning
using GeometricMachineLearning: AutoEncoderLoss, FeedForwardLoss, HNNLoss, LNNLoss,
                                SymplecticEulerLoss, VariationalMidpointLoss
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

include("../helpers/eltypes.jl")

# The four-argument method reconstructs its input and ignores the target, so an
# autoencoder is trained on the input alone. `FeedForwardLoss` on the same four arguments is the
# other value: without it the assertion would hold of any method that ignored its target.
@testset "AutoEncoderLoss reconstructs its input ($T)" for T in REAL_ELTYPES
    Random.seed!(21)
    model = Chain(Dense(4, 2, tanh), Dense(2, 4, tanh))
    nn = NeuralNetwork(model, CPU(), T)
    ps = params(nn)
    x = rand(T, 4, 3)
    y = rand(T, 4, 3)

    loss = AutoEncoderLoss()
    @test loss(nn, x, y) == loss(nn, x)
    @test loss(model, ps, x, y) == loss(model, ps, x)
    @test loss(model, ps, x, y) != FeedForwardLoss()(model, ps, x, y)
    @test loss(model, ps, x) isa T
    @test loss(model, ps, x, y) isa T
end

# A loss's `ps` is a `NetworkParameters`, so a bare `NamedTuple` reaches no method of a
# GML loss rather than a silently different gradient: which of the two AD hands the loss is what an
# untyped argument hides. The four-argument methods of a training loss carry the same annotation,
# and their model slot is reached by `AbstractNeuralNetworks`' own untyped fallback where the wrapper
# is missing.
#
# Not numeric: the element type of the data does not reach the method selection, so this is not run
# over `REAL_ELTYPES`.
@testset "a loss takes a NetworkParameters" begin
    Random.seed!(23)
    hnn_arch = StandardHamiltonianArchitecture(2, 4, 1)
    lnn_arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)

    hnn_input, hnn_output = randn(2, 3), randn(2, 3)
    lnn_input, lnn_output = randn(2, 3), randn(1, 3)

    losses = (
        (HNNLoss(hnn_arch), hnn_arch, hnn_input, hnn_output),
        (LNNLoss(lnn_arch), lnn_arch, lnn_input, lnn_output),
        (SymplecticEulerLoss(hnn_arch, 0.1), hnn_arch, hnn_input, hnn_output),
        (VariationalMidpointLoss(lnn_arch, 0.05), lnn_arch, lnn_input, lnn_output)
    )
    for (loss, arch, input, output) in losses
        nn = NeuralNetwork(arch, CPU(), Float64)
        ps = params(nn)
        # the wrapper is accepted, and the same number comes back through the `NeuralNetwork`
        # method, which forwards to the wrapper
        @test loss(nn, input, output) == loss(ps, input, output)
        # the bare `NamedTuple` under the wrapper is not
        @test_throws MethodError loss(params(ps), input, output)
    end

    # `TransformerLoss` has no three-argument method, so its four-argument ones are what the
    # annotation reaches: the wrapper is accepted, and a bare `NamedTuple` under it reaches no method
    # of this package -- `AbstractNeuralNetworks`' own untyped fallback refuses it.
    transformer = StandardTransformerIntegrator(4; n_heads = 2, L = 1, n_blocks = 1)
    tr = NeuralNetwork(transformer, CPU(), Float64)
    tr_input, tr_output = randn(4, 3, 2), randn(4, 2, 2)
    tr_loss = TransformerLoss(3, 2)
    @test tr_loss(tr.model, tr.params, tr_input, tr_output) ==
          tr_loss(tr, tr_input, tr_output)
    @test_throws ErrorException tr_loss(
        tr.model, params(tr.params), tr_input, tr_output)
end
