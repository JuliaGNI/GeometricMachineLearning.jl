# The optimizer of a training loop is `GeometricOptimizers.TrainingOptimizer`: one cache and one
# state over the whole parameter set, or one per leaf for a `CompositeMethod`, stepped with the
# gradient of one minibatch at a fixed or scheduled step size. What this package adds is the
# `Optimizer` that pairs one with a network, and the epoch loop in `data_loader/optimize.jl` that
# calls it.
#
# There is no step here, no cache, no state and no list of methods. Each of those used to be written
# here a second time -- a per-layer walk, a Euclidean Adam of its own, a chain of `isa` tests over
# the state types -- and each could disagree with `GeometricOptimizers` without a test noticing.

"""
    Optimizer(method, nn; retraction = Cayley(), step_size = default_step_size(method))
    Optimizer(nn; algorithm, linesearch = default_step_size(algorithm), retraction = Cayley())

The optimizer of a training loop for the parameters of `nn`, a `NeuralNetwork` or a
`NetworkParameters`.

It holds one `GeometricOptimizers.TrainingOptimizer` over the whole parameter set, which
[`optimization_step!`](@ref) steps once per minibatch. `method` is a first-order method:
`GradientMethod()`, `MomentumMethod()`, `Adam()`, `ScalarMomentAdam()` or another member of
GeometricOptimizers' `AdamFamily`, or a `CompositeMethod`, which gives every leaf of the set the
method it selects for that leaf:

```julia
method = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
opt = Optimizer(method, nn)
```

`step_size` is a number — a fixed learning rate — or a `DecayingStatic`, a learning rate that decays
geometrically with the iteration number. The keyword form takes the `(algorithm, linesearch)` pair
that `AdamOptimizerWithDecay` returns, so it splats straight in:

```julia
opt = Optimizer(nn; AdamOptimizerWithDecay(n_epochs)...)
```

`retraction` is `Cayley()` or `Geodesic()`.

# Extended help

The step size is a property of the optimizer and not of the method: the same `Adam()` trains at any
learning rate. That is the split `GeometricOptimizers` makes, where the method supplies a direction
and the step size says how far to go along it. A training loop has no objective for a line search to
evaluate, only the loss of one minibatch, so a step size is a number or a schedule and nothing else.
"""
struct Optimizer{TO <: GeometricOptimizers.TrainingOptimizer}
    training::TO
end

function Optimizer(method::OptimizerMethod, nn::NeuralNetwork; kwargs...)
    Optimizer(method, params(nn); kwargs...)
end

function Optimizer(method::OptimizerMethod, ps::NetworkParameters;
        retraction::GeometricOptimizers.AbstractRetraction = Cayley(),
        step_size = GeometricOptimizers.default_step_size(method))
    Optimizer(TrainingOptimizer(ps; algorithm = method, linesearch = step_size,
        retraction = retraction))
end

function Optimizer(nn_or_ps::Union{NeuralNetwork, NetworkParameters};
        algorithm::OptimizerMethod,
        linesearch = GeometricOptimizers.default_step_size(algorithm),
        retraction::GeometricOptimizers.AbstractRetraction = Cayley())
    Optimizer(algorithm, nn_or_ps; retraction = retraction, step_size = linesearch)
end

"""
    optimization_step!(ps, opt::Optimizer, dp)

Take one step of the [`Optimizer`](@ref) `opt` with the gradient `dp` of the loss at the parameters
`ps`, and write the new parameters into `ps`.

`ps` is the parameter set `opt` was built for. `dp` is the gradient a pullback returns, a
`NetworkParameters` or the bare `NamedTuple` of the same shape. This is
`GeometricOptimizers.optimization_step!` on the `TrainingOptimizer` that `opt` holds; see there for
what a step does and what it refuses.
"""
function optimization_step!(ps::NetworkParameters, opt::Optimizer, dp)
    optimization_step!(ps, opt.training, _as_parameter_set(dp))
    ps
end

# A pullback seeded with a bare `NamedTuple` hands its gradient back as one; see `_get_contents`.
_as_parameter_set(dp::NetworkParameters) = dp
_as_parameter_set(dp::NamedTuple) = NetworkParameters(dp)

check(::Optimizer) = nothing
