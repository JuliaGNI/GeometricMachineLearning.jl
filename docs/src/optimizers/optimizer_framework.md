# Neural Network Optimizers

During *optimization* we aim at changing the neural network parameters in such a way to minimize the loss function. So if we express the loss function ``L`` as a function of the neural network weights ``\Theta`` in a parameter space ``\mathbb{P}`` we can phrase the task as: 

```@eval
Main.definition(raw"Given a neural network ``\mathcal{NN}`` parametrized by ``\Theta`` and a loss function ``L:\mathbb{P}\to\mathbb{R}`` we call an algorithm an **iterative optimizer** (or simply **optimizer**) if it performs the following task:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Theta \leftarrow \mathtt{Optimizer}(\Theta, \mathrm{past history}, t),
" * Main.indentation * raw"```
" * Main.indentation * raw"with the aim of decreasing the value ``L(\Theta)`` in each optimization step.")
```

The past history of the optimization is stored in a cache ([`AdamCache`](@ref), [`MomentumCache`](@ref), [`GradientCache`](@ref) etc.) in `GeometricMachineLearning`.

Optimization for neural networks is (almost always) some variation on gradient descent. The most basic form of gradient descent is a discretization of the *gradient flow equation*:

```math
\dot{\theta} = -\nabla_\Theta{}L,
```
by means of a Euler time-stepping scheme: 
```math
\Theta^{t+1} = \Theta^{t} - h\nabla_{\Theta^{t}}L,
```
where ``\eta`` (the time step of the Euler scheme) is referred to as the *learning rate*. 

This equation can easily be generalized to [manifolds](@ref "(Matrix) Manifolds") by replacing the *Euclidean gradient* ``\nabla_{\Theta^{t}}L`` by a *Riemannian gradient* $-h\mathrm{grad}_{\theta^{t}}L$ and addition by $-h\nabla_{\theta^{t}}L$ with a [retraction](../optimizers/manifold_related/retractions.md) by $-h\mathrm{grad}_{\theta^{t}}L$.

## Generalization

In order to generalize neural network optimizers to [homogeneous spaces](@ref "Homogeneous Spaces"), a class of manifolds we often encounter in machine learning, we have to find a [global tangent space representation](@ref "Global Tangent Spaces") which we call $\mathfrak{g}^\mathrm{hor}$ here. 

Starting from an element of the tangent space $T_Y\mathcal{M}$[^1], we need to perform two mappings to arrive at $\mathfrak{g}^\mathrm{hor}$, which we refer to by $\Omega$ and a red horizontal arrow:

[^1]: In practice this is obtained by first using an AD routine on a loss function $L$, and then computing the Riemannian gradient based on this. See the section of the [Stiefel manifold](@ref "The Stiefel Manifold") for an example of this.

```@example
Main.include_graphics("../tikz/general_optimization_with_boundary") # hide
```

Here the mapping $\Omega$ is a [horizontal lift](manifold_related/horizontal_lift.md) from the tangent space onto the **horizontal component of the Lie algebra at $Y$**. 

The red line maps the horizontal component at $Y$, i.e. $\mathfrak{g}^{\mathrm{hor},Y}$, to the horizontal component at $\mathfrak{g}^\mathrm{hor}$.

The $\mathrm{cache}$ stores information about previous optimization steps and is dependent on the optimizer. The elements of the $\mathrm{cache}$ are also in $\mathfrak{g}^\mathrm{hor}$. Based on this the optimer ([Adam](optimizers/adam_optimizer.md) in this case) computes a final velocity, which is the input of a [retraction](optimizers/manifold_related/retractions.md). Because this *update* is done for $\mathfrak{g}^{\mathrm{hor}}\equiv{}T_Y\mathcal{M}$, we still need to perform a mapping, called `apply_section` here, that then finally updates the network parameters. The two red lines are described in [global sections](@ref "Global Sections").

## Library Functions

```@docs; canonical = false
Optimizer
```

## References 

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
```