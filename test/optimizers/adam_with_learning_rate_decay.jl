using GeometricMachineLearning
using GeometricMachineLearning: ResNetLayer, params
using GeometricOptimizers: optimization_step!
using LinearAlgebra: I
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(123)

const sin_vector = sin.(0:0.1:2π)
const dl = DataLoader(reshape(sin_vector, 1, length(sin_vector), 1))

function setup_network(dl::DataLoader{T}) where {T}
    arch = Chain(Dense(1, 20, tanh), ResNetLayer(20, tanh), Dense(20, 1, identity))
    NeuralNetwork(arch, CPU(), T)
end

# tests checks if Adam with decay achieves a lower loss value than regular Adam and the two converge reasonably well
function train_network(; n_epochs = 2048)
    # Seeded per call, as in `svd_optim.jl`: three of these run off one top-level seed, so
    # without it each one starts from whatever RNG state its predecessor left behind.
    Random.seed!(123)
    nn₁ = setup_network(dl)
    nn₂ = setup_network(dl)

    o₁ = TrainingOptimizer(nn₁.params; algorithm = Adam())
    # `AdamOptimizerWithDecay` is GeometricOptimizers', and returns the `(algorithm, linesearch)`
    # pairing the keywords take, so it splats rather than being passed positionally. Its argument
    # counts optimization steps, and the optimizer converts it to the element type of the network.
    o₂ = TrainingOptimizer(nn₂.params; AdamOptimizerWithDecay(n_epochs)...)

    batch = Batch(5, 1)
    loss = FeedForwardLoss()

    loss_array₁ = train!(nn₁, o₁, dl, batch, n_epochs; loss, show_progress = false)
    loss_array₂ = train!(nn₂, o₂, dl, batch, n_epochs; loss, show_progress = false)

    T = eltype(dl)
    @test loss_array₂[end] < loss_array₁[end] < T(1.6e-1)
end

@doc raw"""
`AdamOptimizerWithDecay` also has to work with manifold weights.

The pairing is `Adam` plus a `DecayingStatic` step size, so the method the optimizer dispatches on is
an ordinary `Adam` and the manifold weights take GeometricOptimizers' Adam cache like any other Adam.
"""
# `n_epochs = 128` and not 32: `AdamOptimizerWithDecay(n_epochs)` fixes
# γ = exp(log(η₂/η₁)/n_epochs), so a 32-epoch budget drives the learning rate to η₂ = 1e-6 almost
# at once and the run barely trains -- from a seeded start the loss fell by under 2% on 1.13 and
# *rose* on 1.12. Over 128 epochs the decay is gentle enough that it falls by a factor of six.
function train_manifold_network(; n_epochs = 128)
    Random.seed!(123)
    arch = Chain(StiefelLayer(1, 20), Dense(20, 20, tanh), Dense(20, 1, identity))
    nn = NeuralNetwork(arch, CPU(), eltype(dl))

    o = TrainingOptimizer(nn.params; AdamOptimizerWithDecay(n_epochs)...)
    loss_array = train!(nn, o, dl, Batch(5, 1), n_epochs; loss = FeedForwardLoss(),
        show_progress = false)

    @test all(isfinite, loss_array)
    @test loss_array[end] < loss_array[1]
    # the Stiefel weight has to still be on the manifold
    Y = params(nn).L1.weight
    @test Y' * Y ≈ I
end

@doc raw"""
The schedule is walked from ``t = 1``, not from ``t = 0``.

The step increments the iteration number *before* it reads the step size, so the first step of a run
takes ``\alpha(1) = \gamma\eta_1`` and not ``\alpha(0) = \eta_1``. That is how `DecayingStatic`
counts, because `GeometricOptimizers.solve!` calls `increase_iteration_number!` before
`solver_step!`.
"""
function schedule_starts_at_one(T; n_epochs = 100, η₁ = 1e-2, η₂ = 1e-6)
    γ = exp(log(η₂ / η₁) / n_epochs)

    # One step of a Euclidean parameter with a gradient of `1` moves it by `α₁ / (√1 + δ) ≈ α₁`, so
    # the distance travelled reports the α used.
    ps = NetworkParameters((weight = zeros(T, 2, 2),))
    opt = TrainingOptimizer(ps; AdamOptimizerWithDecay(n_epochs; η₁, η₂)...)
    @test opt.linesearch isa DecayingStatic
    @test opt.state.iterations == 0
    optimization_step!(ps, opt, NetworkParameters((weight = ones(T, 2, 2),)))
    @test opt.state.iterations == 1
    @test eltype(ps.weight) === T
    @test abs(ps.weight[1, 1]) ≈ η₁ * γ rtol = 1e-6
end

train_network()
train_manifold_network()
@testset "the decay schedule starts at t = 1 ($T)" for T in REAL_ELTYPES
    schedule_starts_at_one(T)
end
