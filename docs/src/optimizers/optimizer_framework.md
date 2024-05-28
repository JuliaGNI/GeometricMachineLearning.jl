# Neural Network Optimizers

In this section we present the general Optimizer framework used in `GeometricMachineLearning`. For more information on the particular steps involved in this consult the documentation on the various optimizer methods such as the *momentum optimizer* and the [Adam optimizer](@ref "The Adam Optimizer"), and the documentation on [retractions](@ref "Retractions").

During *optimization* we aim at changing the neural network parameters in such a way to minimize the loss function. So if we express the loss function ``L`` as a function of the neural network weights ``\Theta`` in a parameter space ``\mathbb{P}`` we can phrase the task as: 

```@eval
Main.definition(raw"Given a neural network ``\mathcal{NN}`` parametrized by ``\Theta`` and a loss function ``L:\mathbb{P}\to\mathbb{R}`` we call an algorithm an **iterative optimizer** (or simply **optimizer**) if it performs the following task:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Theta \leftarrow \mathtt{Optimizer}(\Theta, \text{past history}, t),
" * Main.indentation * raw"```
" * Main.indentation * raw"with the aim of decreasing the value ``L(\Theta)`` in each optimization step.")
```

The past history of the optimization is stored in a cache ([`AdamCache`](@ref), [`MomentumCache`](@ref), [`GradientCache`](@ref) etc.) in `GeometricMachineLearning`.

Optimization for neural networks is (almost always) some variation on gradient descent. The most basic form of gradient descent is a discretization of the *gradient flow equation*:

```math
\dot{\Theta} = -\nabla_\Theta{}L,
```
by means of an Euler time-stepping scheme: 
```math
\Theta^{t+1} = \Theta^{t} - h\nabla_{\Theta^{t}}L,
```
where ``\eta`` (the time step of the Euler scheme) is referred to as the *learning rate*. 

This equation can easily be generalized to [manifolds](@ref "(Matrix) Manifolds") by replacing the *Euclidean gradient* ``\nabla_{\Theta^{t}}L`` by a *Riemannian gradient* ``-h\mathrm{grad}_{\Theta^{t}}L`` and addition by ``-h\nabla_{\Theta^{t}}L`` with the [exponential map](@ref "Geodesic Sprays and the Exponential Map") of ``-h\mathrm{grad}_{\theta^{t}}L``. In practice we often use approximations ot the exponential map however. These are called [retractions](@ref "Retractions").

## Generalization to Homogeneous Spaces

In order to generalize neural network optimizers to [homogeneous spaces](@ref "Homogeneous Spaces") we utilize their corresponding [global tangent space representation](@ref "Global Tangent Spaces") ``\mathfrak{g}^\mathrm{hor}``. 

When introducing the notion of a [global tangent space](@ref "Global Tangent Spaces") we discussed how an element of the tangent space ``T_Y\mathcal{M}`` can be represented in ``\mathfrak{g}^\mathrm{hor}`` by performing two mappings: the first one is the horizontal lift ``\Omega`` (see the docstring for [`GeometricMachineLearning.Ω`](@ref)) and the second one is the adjoint operation[^1] with the lift of ``Y`` called ``\lambda(Y)``. We can visualize the steps required in performing this generalization:

[^1]: By the *adjoint operation* ``\mathrm{ad}_A:\mathfrak{g}\to\mathfrak{g}`` for an element ``A\in{}G`` we mean ``B \mapsto A^{-1}BA``.

```@example
Main.include_graphics("../tikz/general_optimization_with_boundary") # hide
```

The `cache` stores information about previous optimization steps and is dependent on the optimizer. In general the cache is represented as one or more elements in ``\mathfrak{g}^\mathrm{hor}``. Based on this the optimizer method (represented by [`update!`](@ref) in the figure) computes a *final velocity*. This final velocity is again an element of ``\mathfrak{g}^\mathrm{hor}``.

The final velocity is then fed into a [retraction](@ref "Retractions")[^2]. For computational reasons we split the retraction into two steps, referred to as "Retraction" and [`apply_section`](@ref) above. These two mappings together are equivalent to: 

[^2]: A retraction is an approximation of the [exponential map](@ref "Geodesic Sprays and the Exponential Map")

```math
\mathrm{retraction}(\Delta) = \mathrm{retraction}(\lambda(Y)B^\Delta{}E) = \lambda(Y)\mathrm{Retraction}(B^\Delta), 
```

where ``\Delta\in{}T_\mathcal{M}`` and ``B^\Delta`` is its representation in ``\mathfrak{g}^\mathrm{hor}`` as ``B^\Delta = \lambda(Y)^{-1}\Omega(\Delta)\lambda(Y).``


## Library Functions

```@docs; canonical = false
Optimizer
update!
```

## References 

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
```