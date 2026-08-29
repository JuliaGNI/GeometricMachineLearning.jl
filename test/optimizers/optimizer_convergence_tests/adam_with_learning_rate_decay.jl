
using GeometricMachineLearning
using GeometricMachineLearning: ResNetLayer, params
using LinearAlgebra: I
using Test
import Random

Random.seed!(123)

const sin_vector = sin.(0:0.1:2π)
const dl = DataLoader(reshape(sin_vector, 1, length(sin_vector), 1))

function setup_network(dl::DataLoader{T}) where T
    arch = Chain(Dense(1, 20, tanh), ResNetLayer(20, tanh), Dense(20, 1, identity))
    NeuralNetwork(arch, CPU(), T)
end

# tests checks if Adam with decay achieves a lower loss value than regular Adam and the two converge reasonably well
function train_network(; n_epochs=2048)
    # Seeded per call, as in `svd_optim.jl`: three of these run off one top-level seed, so
    # without it each one starts from whatever RNG state its predecessor left behind.
    Random.seed!(123)
    nn₁ = setup_network(dl)
    nn₂ = setup_network(dl)

    o₁ = Optimizer(AdamOptimizer(), nn₁)
    # `AdamOptimizerWithDecay` is GeometricOptimizers' now, and returns the `(algorithm, linesearch)`
    # pairing its own `Optimizer` takes, so it splats rather than being passed positionally. The
    # element type is positional and defaults to `Float64` here, where GML's own version took it
    # from `η₁` and so defaulted to `Float32`.
    o₂ = Optimizer(nn₂; AdamOptimizerWithDecay(n_epochs, eltype(dl))...)

    batch = Batch(5, 1)
    loss = FeedForwardLoss()

    loss_array₁ = o₁(nn₁, dl, batch, n_epochs, loss; show_progress = false)
    loss_array₂ = o₂(nn₂, dl, batch, n_epochs, loss; show_progress = false)

    T = eltype(dl)
    @test loss_array₂[end] < loss_array₁[end] < T(1.6e-1)
end

@doc raw"""
`AdamOptimizerWithDecay` also has to work with manifold weights.

The pairing is `Adam` plus a `DecayingStatic` step size, so the method the optimizer dispatches on is
an ordinary `Adam` and the manifold weights take GeometricOptimizers' Adam cache like any other Adam.
GML's own version of this method was a distinct `OptimizerMethod` that had to be routed onto that
cache explicitly, and without the routing every weight -- the `StiefelManifold` ones included -- fell
through to the Euclidean state, whose zero element is a `StiefelLieAlgHorMatrix` and not a manifold
point.
"""
# `n_epochs = 128` and not 32: `AdamOptimizerWithDecay(n_epochs)` fixes
# γ = exp(log(η₂/η₁)/n_epochs), so a 32-epoch budget drives the learning rate to η₂ = 1e-6 almost
# at once and the run barely trains -- from a seeded start the loss fell by under 2% on 1.13 and
# *rose* on 1.12. Over 128 epochs the decay is gentle enough that it falls by a factor of six.
function train_manifold_network(; n_epochs = 128)
    Random.seed!(123)
    arch = Chain(StiefelLayer(1, 20), Dense(20, 20, tanh), Dense(20, 1, identity))
    nn = NeuralNetwork(arch, CPU(), eltype(dl))

    o = Optimizer(nn; AdamOptimizerWithDecay(n_epochs, eltype(dl))...)
    loss_array = o(nn, dl, Batch(5, 1), n_epochs, FeedForwardLoss(); show_progress = false)

    @test all(isfinite, loss_array)
    @test loss_array[end] < loss_array[1]
    # the Stiefel weight has to still be on the manifold
    Y = params(nn).L1.weight
    @test Y' * Y ≈ I
end

@doc raw"""
The schedule is walked from ``t = 1``, not from ``t = 0``.

`optimization_step!` increments `opt.iterations` *before* it reads the step size, so the first step
of a run takes ``\alpha(1) = \gamma\eta_1`` and not ``\alpha(0) = \eta_1``. That is how the pre-0.5
`AdamOptimizerWithDecay` counted — it incremented `o.step` before `update!` — and how
`DecayingStatic` counts, because `GeometricOptimizers.solve!` calls `increase_iteration_number!`
before `solver_step!`. Reading before incrementing put every step of a run one place early in the
schedule, which `test/adam_optimizer_with_decay.jl` upstream asserts does not happen.
"""
function schedule_starts_at_one(; n_epochs = 100, η₁ = 1e-2, η₂ = 1e-6)
    method = AdamOptimizerWithDecay(n_epochs, Float64; η₁ = η₁, η₂ = η₂)
    o = Optimizer(NetworkParameters((weight = zeros(2, 2),)); method...)

    γ = exp(log(η₂ / η₁) / n_epochs)
    @test o.step_size isa DecayingStatic
    @test o.iterations == 0

    for t in 1:4
        o.iterations += 1
        @test GeometricMachineLearning._current_step_size(o, o.iterations) ≈ η₁ * γ^t
    end

    # and the same thing through the public entry point: one step of a Euclidean parameter with a
    # gradient of `1` moves it by `α₁ / (√1 + δ) ≈ α₁`, so the distance travelled reports the α used
    ps = NetworkParameters((weight = zeros(2, 2),))
    opt = Optimizer(ps; method...)
    optimization_step!(opt, GlobalSection(ps), ps, (weight = ones(2, 2),))
    @test opt.iterations == 1
    @test abs(ps.weight[1, 1]) ≈ η₁ * γ rtol = 1e-6
end

train_network()
train_manifold_network()
schedule_starts_at_one()
