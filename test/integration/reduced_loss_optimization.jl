using GeometricMachineLearning
using GeometricMachineLearning: ReducedLoss
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(123)

# `ReducedLoss` trained through `train!`, which is the path the symplectic autoencoder tutorial
# documents and the path `scripts/reproduction/symplectic_autoencoders/` takes. The loss annotates
# its parameter argument `::NetworkParameters`, and NeuralNetworkParameters 0.5's `Zygote` rule keeps
# the wrapper on the forward pass, so the annotation is dispatchable under AD. The docstring example
# calls the loss directly, which dispatches, and the tutorial's training is in a plain ```julia fence
# that Documenter renders and never executes; this testset is the one that runs the training.
@testset "ReducedLoss trains through train! ($T)" for T in REAL_ELTYPES
    reduced_dim = 2

    q = rand(T, 6, 30, 2)
    p = rand(T, 6, 30, 2)
    dl = DataLoader((q = q, p = p); autoencoder = true)

    autoencoder = NeuralNetwork(SymplecticAutoencoder(dl.input_dim, reduced_dim), CPU(), T)
    loss = ReducedLoss(encoder(autoencoder), decoder(autoencoder))

    # The data the reduced integrator sees: one tensor rather than a `(q, p)` pair.
    time_series = DataLoader(dl; autoencoder = false)
    dl_reduced = DataLoader(vcat(time_series.input.q, time_series.input.p))

    integrator = NeuralNetwork(
        StandardTransformerIntegrator(reduced_dim; transformer_dim = 10, n_blocks = 1,
            n_heads = 2, L = 1, upscaling_activation = tanh), CPU(), T)

    o = TrainingOptimizer(integrator.params; algorithm = Adam())
    batch = Batch(8, 4)

    # The parameters reach the loss through `Zygote.pullback`, which under NeuralNetworkParameters
    # 0.5 keeps the `NetworkParameters` the loss annotates.
    loss_array = train!(integrator, o, dl_reduced, batch, 10; loss, show_progress = false)

    @test eltype(loss_array) === T
    @test length(loss_array) == 10
    @test all(isfinite, loss_array)
    @test loss_array[end] < loss_array[1]
end
