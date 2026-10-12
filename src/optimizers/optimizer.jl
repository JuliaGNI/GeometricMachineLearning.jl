"""
    NetworkOptimizer(nn; algorithm = Adam(), linesearch = default_step_size(algorithm), retraction = Cayley())
    NetworkOptimizer(ps; algorithm, linesearch, retraction)

The optimizer that [`train!`](@ref) trains the neural network `nn`, or the parameters `ps`, with.

It holds one `GeometricOptimizers.TrainingOptimizer` over the whole parameter set, and the keywords
are that constructor's: `algorithm` is a first-order method such as `GradientMethod()`,
`MomentumMethod()` or `Adam()`, `linesearch` is a fixed step size or a `DecayingStatic` schedule,
and `retraction` is `Cayley()` or `Geodesic()`. `AdamOptimizerWithDecay(n)` gives `algorithm` and
`linesearch` together, and its `n` counts optimization steps, one per minibatch:

```julia
opt = NetworkOptimizer(nn; AdamOptimizerWithDecay(n_steps)...)
```

The optimizer carries the iteration number, the moments and the global sections from one call of
[`train!`](@ref) to the next, so a second call resumes the training. It is built for the parameters
of one network, and it is not checked when it is used with another.

A loop of one's own takes a step with `GeometricOptimizers.optimization_step!(params(nn), opt, dp)`,
where `dp` is the gradient of the loss as a `NetworkParameters` of the shape of `params(nn)`.
"""
struct NetworkOptimizer{OT <: TrainingOptimizer}
    optimizer::OT
end

function NetworkOptimizer(ps::NetworkParameters; kwargs...)
    NetworkOptimizer(TrainingOptimizer(ps; kwargs...))
end
NetworkOptimizer(nn::NeuralNetwork; kwargs...) = NetworkOptimizer(params(nn); kwargs...)

function optimization_step!(ps, opt::NetworkOptimizer, dp)
    optimization_step!(ps, opt.optimizer, dp)
end
