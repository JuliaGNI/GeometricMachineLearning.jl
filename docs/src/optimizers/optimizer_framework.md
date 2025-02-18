```@raw latex
In this chapter we introduce a \textit{general framework for manifold optimization} that is needed to efficiently train symplectic autoencoders. We start this chapter by discussing optimization for neural network in general and explain how we can generalize this to homogeneous spaces. We will see that an important ingredient for doing so are \textit{retractions} which we then elaborate on. After discussing how to make the computation of retractions efficient for homogeneous spaces we conclude the chapter by introducing the notion of \textit{parallel transport} which we need to extend the notion of \textit{momentum} in neural network optimization.
```

# Neural Network Optimizers

In this section we present the general Optimizer framework used in `GeometricMachineLearning`. For more information on the particular steps involved in this consult the documentation on the various optimizer methods such as the gradient optimizer, the momentum optimizer and the [Adam optimizer](@ref "The Adam Optimizer"), and the documentation on [retractions](@ref "Retractions").

During *optimization* we aim at changing the neural network parameters in such a way to minimize the loss function. A loss function assigns a scalar value to the weights that [parametrize the neural network](@ref "Structure-Preserving Neural Networks"):

```math
    L: \mathbb{P}\to\mathbb{R}_{\geq0},\quad \Theta \mapsto L(\Theta),
```

where ``\mathbb{P}`` is the parameter space. We can then phrase the optimization task as: 

```@eval
Main.definition(raw"Given a neural network ``\mathcal{NN}`` parametrized by ``\Theta`` and a loss function ``L:\mathbb{P}\to\mathbb{R}`` we call an algorithm an **iterative optimizer** (or simply **optimizer**) if it performs the following task:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Theta \leftarrow \mathtt{Optimizer}(\Theta, \text{past history}, t),
" * Main.indentation * raw"```
" * Main.indentation * raw"with the aim of decreasing the value ``L(\Theta)`` in each optimization step.")
```

The past history of the optimization is stored in a cache ([`AdamCache`](@ref), [`MomentumCache`](@ref), [`GradientCache`](@ref), ``\ldots``) in `GeometricMachineLearning`.

Optimization for neural networks is (almost always) some variation on gradient descent. The most basic form of gradient descent is a discretization of the *gradient flow equation*:

```math
\dot{\Theta} = -\nabla_\Theta{}L,
```
by means of an Euler time-stepping scheme: 
```math
\Theta^{t+1} = \Theta^{t} - h\nabla_{\Theta^{t}}L,
```
where ``\eta`` (the time step of the Euler scheme) is referred to as the *learning rate*. 

This equation can easily be generalized to [manifolds](@ref "(Matrix) Manifolds") with the following two steps:
1. modify ``-\nabla_{\Theta^{t}}L\implies{}-h\mathrm{grad}_{\Theta^{t}}L,`` i.e. replace the Euclidean gradient by a [Riemannian gradient](@ref "The Riemannian Gradient") and
2. replace addition with the [geodesic map](@ref "Geodesic Sprays and the Exponential Map").

To sum up, we then have:

```math
\Theta^{t+1} = \mathrm{geodesic}(\Theta^{t}, -h\mathrm{grad}_{\Theta^{t}}L).
```

In practice we very often do not use the geodesic map but approximations thereof. These approximations are called [retractions](@ref "Retractions").

## Generalization to Homogeneous Spaces

In order to generalize neural network optimizers to [homogeneous spaces](@ref "Homogeneous Spaces") we utilize their corresponding [global tangent space representation](@ref "Global Tangent Spaces") ``\mathfrak{g}^\mathrm{hor}``. 

When introducing the notion of a [global tangent space](@ref "Global Tangent Spaces") we discussed how an element of the tangent space ``T_Y\mathcal{M}`` can be represented in ``\mathfrak{g}^\mathrm{hor}`` by performing two mappings: 
1. the first one is the horizontal lift ``\Omega`` (see the docstring for [`GeometricMachineLearning.Ω`](@ref)) and 
2. the second one is performing the adjoint operation[^1] of ``\lambda(Y),`` the section of ``Y``, on ``\Omega(\Delta).`` 

[^1]: By the *adjoint operation* ``\mathrm{ad}_A:\mathfrak{g}\to\mathfrak{g}`` for an element ``A\in{}G`` we mean ``B \mapsto A^{-1}BA``.

The two steps together are performed as [`global_rep`](@ref) in `GeometricMachineLearning.` So we lift to ``\mathfrak{g}^\mathrm{hor}``:

```math
\mathtt{global\_rep}: T_Y\mathcal{M} \to \mathfrak{g}^\mathrm{hor},
```

and then perform all the steps of the optimizer in ``\mathfrak{g}^\mathrm{hor}.`` We can visualize all the steps required in the generalization of the optimizers:

![](../tikz/general_optimization_with_boundary_light.png)
![](../tikz/general_optimization_with_boundary_dark.png)

This picture summarizes all steps involved in an optimization step:
1. map the Euclidean gradient ``\nabla{}L\in\mathbb{R}^{N\times{}n}`` that was obtained via [automatic differentiation](@ref "Pullbacks and Automatic Differentiation") to the Riemannian gradient ``\mathrm{grad}L\in{}T_Y\mathcal{M}`` with the function [`rgrad`](@ref),
2. obtain the global tangent space representation of ``\mathrm{grad}L`` in ``\mathfrak{g}^\mathrm{hor}`` with the function [`global_rep`](@ref),
3. perform an [`update!`](@ref); this consists of two steps: (i) update the cache and (ii) output a *final velocity*,
4. use this final velocity to update the [global section](@ref "Global Sections") ``\Lambda\in{}G,``
5. use the updated global section to update the neural network weight ``\in\mathcal{M}.`` This is done with [`apply_section`](@ref).

The `cache` stores information about previous optimization steps and is dependent on the optimizer. Typically the cache is represented as one or more elements in ``\mathfrak{g}^\mathrm{hor}``. Based on this the optimizer method (represented by [`update!`](@ref) in the figure) computes a *final velocity*. This final velocity is again an element of ``\mathfrak{g}^\mathrm{hor}``. The particular form of the cache and the updating rule depends on which [optimizer method we use](@ref "Standard Neural Network Optimizers").

The final velocity is then fed into a [retraction](@ref "Retractions")[^2]. For computational reasons we split the retraction into two steps, referred to as "Retraction" and [`apply_section`](@ref) above. These two mappings together are equivalent to: 

[^2]: A retraction is an approximation of the [geodesic map](@ref "Geodesic Sprays and the Exponential Map")

```math
\mathrm{retraction}(\Delta) = \mathrm{retraction}(\lambda(Y)B^\Delta{}E) = \lambda(Y)\mathrm{Retraction}(B^\Delta), 
```

where ``\Delta\in{}T_\mathcal{M}`` and ``B^\Delta`` is its representation in ``\mathfrak{g}^\mathrm{hor}`` as ``B^\Delta = \lambda(Y)^{-1}\Omega(\Delta)\lambda(Y).``


## Library Functions

```@docs
Optimizer
optimize_for_one_epoch!
optimization_step!
```

```@raw latex
\begin{comment}
```

## References 

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
```

```@raw latex
\end{comment}
```