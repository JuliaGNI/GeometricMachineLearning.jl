# Training a Neural Network in `GeometricMachineLearning`

The general framework for optimization on homogeneous spaces — the Riemannian gradient, the lift to
the global tangent space ``\mathfrak{g}^\mathrm{hor}``, the optimizer cache, the retraction and the
global section — belongs to `GeometricOptimizers` and is described in its documentation, under
[Optimization on Homogeneous Spaces](@extref GeometricOptimizers :doc:`manifold_optimizers`) and
[Retractions](@extref GeometricOptimizers :doc:`retractions`). The training step itself is
`GeometricOptimizers`' too: its `TrainingOptimizer` steps a whole parameter set with a gradient the
caller computed, and applies the right update to each leaf — a retraction for a weight on a
manifold, ordinary arithmetic for a Euclidean one.

What `GeometricMachineLearning` adds is the part that is about *neural networks*: a
[`NetworkOptimizer`](@ref) built from a `NeuralNetwork`, and [`train!`](@ref), which draws the
minibatches of a data loader over epochs, computes the gradient of the loss on each with automatic
differentiation, and takes one step per minibatch.

The gradient comes from one minibatch at a time, so there is no objective function to hand a line
search; the step size is a number or a `GeometricOptimizers.DecayingStatic` schedule:

```julia
opt = NetworkOptimizer(nn; algorithm = Adam(), linesearch = 1e-3)
opt = NetworkOptimizer(nn; AdamOptimizerWithDecay(n_steps)...)
history = train!(nn, opt, dl, Batch(batch_size), n_epochs)
```

## Library Functions

```@docs
NetworkOptimizer
train!
```
