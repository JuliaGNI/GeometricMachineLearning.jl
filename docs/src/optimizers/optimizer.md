# The `Optimizer` in `GeometricMachineLearning`

The general framework for optimization on homogeneous spaces — the Riemannian gradient, the lift to
the global tangent space ``\mathfrak{g}^\mathrm{hor}``, the optimizer cache, the retraction and the
global section — belongs to `GeometricOptimizers` and is described in its documentation, under
[Optimization on Homogeneous Spaces](@extref GeometricOptimizers :doc:`manifold_optimizers`) and
[Retractions](@extref GeometricOptimizers :doc:`retractions`).

The training step itself is `GeometricOptimizers`' `TrainingOptimizer` as well: one cache and one
state over the whole parameter set of a network, and a retraction for every weight on a manifold.
What `GeometricMachineLearning` adds is the part that is about *neural networks*: the
[`Optimizer`](@ref) that pairs a `TrainingOptimizer` with a `NeuralNetwork`, and the loop that drives
it from a data loader over epochs and batches.

The gradient comes from automatic differentiation on one batch at a time, so there is no objective
function to hand a line search; the step size is a property of the [`Optimizer`](@ref) instead. It is
either a number, or a `GeometricOptimizers.DecayingStatic` schedule:

```julia
opt = Optimizer(Adam(), nn; step_size = 1e-3)
opt = Optimizer(nn; AdamOptimizerWithDecay(n_epochs)...)
```

A network whose weights are not all of one kind can take a different method for each kind. A
`CompositeMethod` chooses the method per leaf of the parameter set, and every leaf then gets the cache
and the state of the method chosen for it. `ScalarMomentAdam` steps a single `StiefelManifold`, so a
network with Stiefel weights beside Euclidean ones needs a composite to use it:

```julia
opt = Optimizer(CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam()), nn)
```

## Library Functions

```@docs
Optimizer
optimize_for_one_epoch!
optimization_step!(::NetworkParameters, ::Optimizer, ::Any)
```
