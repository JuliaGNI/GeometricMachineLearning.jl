using GeometricMachineLearning
using GeometricMachineLearning: ReducedLoss
using Test
import Random

Random.seed!(123)

# `ReducedLoss` trained through the `Optimizer` functor, which is the path the symplectic
# autoencoder tutorial documents and the path `scripts/reproduction/symplectic_autoencoders/`
# takes. It was unreachable: the loss annotated its parameter argument `::NetworkParameters`, and
# `Zygote.pullback` evaluates the forward pass with the parameters unwrapped -- the closure body
# receives the underlying `NamedTuple` -- so the method did not match under AD and the untyped
# `NetworkLoss` fallback in `AbstractNeuralNetworks` caught the call. That fallback's body is an
# `error`, so the failure read as "Functor not defined for `NetworkLoss`" rather than as a
# `MethodError` naming the argument types.
#
# Nothing had caught it because nothing ran it. The docstring example calls the loss directly,
# which dispatches, and the tutorial's training is in a plain ```julia fence that Documenter
# renders and never executes. Restoring the annotation makes the `Optimizer` call below raise
# that `error`, so this testset errors before any of the three assertions runs; the assertions
# themselves check what the loss is for.
@testset "ReducedLoss trains through the Optimizer functor" begin
    reduced_dim = 2

    q = rand(Float64, 6, 30, 2)
    p = rand(Float64, 6, 30, 2)
    dl = DataLoader((q = q, p = p); autoencoder = true)

    autoencoder = NeuralNetwork(SymplecticAutoencoder(dl.input_dim, reduced_dim), CPU(), Float64)
    loss = ReducedLoss(encoder(autoencoder), decoder(autoencoder))

    # The data the reduced integrator sees: one tensor rather than a `(q, p)` pair.
    time_series = DataLoader(dl; autoencoder = false)
    dl_reduced = DataLoader(vcat(time_series.input.q, time_series.input.p))

    integrator = NeuralNetwork(
        StandardTransformerIntegrator(reduced_dim; transformer_dim = 10, n_blocks = 1,
            n_heads = 2, L = 1, upscaling_activation = tanh), CPU(), Float64)

    o = Optimizer(Adam(), integrator)
    batch = Batch(8, 4)

    # The parameters reach the loss through `Zygote.pullback` here, which is what an annotated
    # `params` argument breaks; calling the loss directly would pass either way.
    loss_array = o(integrator, dl_reduced, batch, 10, loss; show_progress = false)

    @test length(loss_array) == 10
    @test all(isfinite, loss_array)
    @test loss_array[end] < loss_array[1]
end
