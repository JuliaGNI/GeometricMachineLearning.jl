# The `Optimizer` in `GeometricMachineLearning`

The general framework for optimization on homogeneous spaces — the Riemannian gradient, the lift to
the global tangent space ``\mathfrak{g}^\mathrm{hor}``, the optimizer cache, the retraction and the
global section — belongs to `GeometricOptimizers` and is described in its documentation, under
[Optimization on Homogeneous Spaces](@extref GeometricOptimizers :doc:`manifold_optimizers`) and
[Retractions](@extref GeometricOptimizers :doc:`retractions`).

What `GeometricMachineLearning` adds is the part that is about *neural networks*: walking the
parameter tree of a `NeuralNetwork`, applying the right update to each leaf — a retraction
for a weight on a manifold, ordinary arithmetic for a Euclidean one — and driving that from a data
loader over epochs and batches.

The gradient comes from automatic differentiation on one batch at a time, so there is no objective
function to hand a line search; the step size is a property of the [`Optimizer`](@ref) instead. It is
either a number, or a `GeometricOptimizers.DecayingStatic` schedule:

```julia
opt = Optimizer(Adam(Float32), nn; step_size = 1e-3)
opt = Optimizer(nn; AdamOptimizerWithDecay(n_epochs, Float32)...)
```

## Library Functions

```@docs
Optimizer
optimize_for_one_epoch!
optimization_step!
GeometricMachineLearning._as_go_solution
```
