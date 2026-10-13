# Training a Neural Network in `GeometricMachineLearning`

The general framework for optimization on homogeneous spaces — the Riemannian gradient, the lift to
the global tangent space ``\mathfrak{g}^\mathrm{hor}``, the optimizer cache, the retraction and the
global section — belongs to `GeometricOptimizers` and is described in its documentation, under
[Optimization on Homogeneous Spaces](@extref GeometricOptimizers :doc:`manifold_optimizers`) and
[Retractions](@extref GeometricOptimizers :doc:`retractions`). The training step itself is
`GeometricOptimizers`' too: its
[`TrainingOptimizer`](@extref GeometricOptimizers GeometricOptimizers.TrainingOptimizer), which
`GeometricMachineLearning` re-exports, steps a whole parameter set with a gradient the caller
computed, and applies the right update to each leaf — a retraction for a weight on a manifold,
ordinary arithmetic for a Euclidean one. It is built on the parameters of a network, `nn.params`.

What `GeometricMachineLearning` adds is the part that is about *neural networks*: [`train!`](@ref),
which draws the minibatches of a data loader over epochs, computes the gradient of the loss on each
with automatic differentiation or a symbolic pullback, and takes one step per minibatch.

The gradient comes from one minibatch at a time, so there is no objective function to hand a line
search; the step size is a number or a `GeometricOptimizers.DecayingStatic` schedule:

```julia
opt = TrainingOptimizer(nn.params; algorithm = Adam(), linesearch = 1e-3)
opt = TrainingOptimizer(nn.params; AdamOptimizerWithDecay(n_steps)...)
history = train!(nn, opt, dl, Batch(batch_size), n_epochs)
```

## Library Functions

```@docs
train!
```
