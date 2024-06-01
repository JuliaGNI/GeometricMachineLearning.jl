# Standard Neural Network Optimizers

In this section we discuss optimization methods that are often used in training neural networks. The [BFGS optimizer](@ref "The BFGS Optimizer") may also be viewed as a *standard neural network optimizer* but is treated in a separate section because of its complexity. From a perspective of manifolds the *optimizer methods* outlined here operate on ``\mathfrak{g}^\mathrm{hor}`` only. Each of them has a cache associated with it[^1] and this cache is updated with the function [`update!`](@ref). The precise role of this function is described below.

[^1]: In the case of the [gradient optimizer](@ref "The Gradient Optimizer") this cache is trivial.

## The Gradient Optimizer

The gradient optimizer is the simplest optimization algorithm used to train neural networks. It was already briefly discussed when we introduced [Riemannian manifolds](@ref "Gradient Flows and Riemannian Optimization").

It simply does: 

```math
\mathrm{weight} \leftarrow \mathrm{weight} + (-\eta\cdot\mathrm{gradient}),
```

where addition has to be replaced with appropriate operations in the manifold case.

When calling [`GradientOptimizer`](@ref) we can specify a learning rate ``\eta`` (or use the default).

```@example optimizer_methods
using GeometricMachineLearning

const η = 0.01
method = GradientOptimizer(η)
```

In order to use the optimizer we need an instance of [`Optimizer`](@ref) that is called with the method and the weights of the neural network:


```@example optimizer_methods
weight = (A = zeros(10, 10), )
o = Optimizer(method, weight)
```

If we operate on a derivative with [`update!`](@ref) this will compute a *final velocity* that is then used to compute a retraction (or simply perform addition if we do not deal with a manifold):

```@example optimizer_methods
dx = (A = one(weight.A), )
update!(o, o.cache, dx)

dx
```

So what has happened here is that the gradient `dx` was simply multiplied with ``-\eta`` as the cache of the gradient optimizer is trivial.

## The Momentum Optimizer

The momentum optimizer is similar to the gradient optimizer but further stores past information as *first moments*. We let these first moments *decay* with a *decay parameter* ``\alpha``:

```math
\mathrm{weights} \leftarrow weights + (\alpha\cdot\mathrm{moment} - \eta\cdot\mathrm{gradient}),
```

where addition has to be replaced with appropriate operations in the manifold case.

In the case of the momentum optimizer the cache is non-trivial:

```@example optimizer_methods
const α = 0.5
method = MomentumOptimizer(η, α)
o = Optimizer(method, weight)

o.cache
```

But as the cache is initialized with zeros it will lead the same result as the gradient optimizer in the first iteration:

```@example optimizer_methods
dx = (A = one(weight.A), )

update!(o, o.cache, dx)

dx
```

The cache has changed however:

```@example optimizer_methods
o.cache
```

If we have weights on manifolds calling [`Optimizer`](@ref) will automatically allocate the correct cache on ``\mathfrak{g}^\mathrm{hor}``:

```@example optimizer_methods
weight = (Y = rand(StiefelManifold, 10, 5), )

Optimizer(method, weight).cache
```

## The Adam Optimizer 

The Adam Optimizer is one of the most widely neural network optimizer. Like most modern neural network optimizers it contains a `cache` that is updated based on first-order gradient information and then, in a second step, the `cache` is used to compute a velocity estimate for updating the neural network weights. 

Here we first describe the Adam algorithm for the case where all the weights are on a vector space and then show how to generalize this to the case where the weights are on a manifold. 

### All weights on a vector space

The cache of the Adam optimizer consists of *first and second moments*. The *first moments* ``B_1``, similar to the momentum optimizer, store linear information about the current and previous gradients, and the *second moments* ``B_2`` store quadratic information about current and previous gradients (all computed from a first-order gradient). 

If all the weights are on a vector space, then we directly compute updates for ``B_1`` and ``B_2``:
1. ``B_1 \gets ((\rho_1 - \rho_1^t)/(1 - \rho_1^t))\cdot{}B_1 + (1 - \rho_1)/(1 - \rho_1^t)\cdot{}\nabla{}L,``
2. ``B_2 \gets ((\rho_2 - \rho_1^t)/(1 - \rho_2^t))\cdot{}B_2 + (1 - \rho_2)/(1 - \rho_2^t)\cdot\nabla{}L\odot\nabla{}L,``

where ``\odot:\mathbb{R}^n\times\mathbb{R}^n\to\mathbb{R}^n`` is the *Hadamard product*: ``[a\odot{}b]_i = a_ib_i$. $\rho_1`` and $\rho_2$ are hyperparameters. Their defaults, $\rho_1=0.9$ and $\rho_2=0.99$, are taken from [goodfellow2016deep; page 301](@cite). After having updated the `cache` (i.e. $B_1$ and $B_2$) we compute a **velocity** (step 3) with which the parameters $Y_t$ are then updated (step 4).

3. $W_t\gets -\eta{}B_1/\sqrt{B_2 + \delta},$
4. $Y_{t+1} \gets Y_t + W_t,$

Here $\eta$ (with default 0.01) is the **learning rate** and $\delta$ (with default $3\cdot10^{-7}$) is a small constant that is added for stability. The division, square root and addition in step 3 are performed element-wise. 

```@example 
Main.include_graphics("../tikz/adam_optimizer") # hide
```

```@eval
Main.remark("The optimization framework presented here manages to generalize the Adam optimizer to manifolds without knowing an underlying differential equation. From a mathematical perspective this is not really satisfactory because we would ideally want the optimizers to emerge as a discretization of a differential equation as in the case of the gradient and the momentum optimizer to better interpret them. A similar attempt to generalize Adam to the Stiefel manifold was made in [kong2023momentum](@cite).")
```

## Weights on manifolds 

The problem with generalizing Adam to manifolds is that the Hadamard product $\odot$ as well as the other element-wise operations ($/$, $\sqrt{}$ and $+$ in step 3 above) lack a clear geometric interpretation. In `GeometricMachineLearning` we get around this issue by utilizing a so-called [global tangent space representation](@ref "Global Tangent Spaces").  

## The Adam Optimizer with Decay


## Library Functions

```@docs; canonical=false
OptimizerMethod
GradientOptimizer
MomentumOptimizer
AdamOptimizer
AdamOptimizerWithDecay
GeometricMachineLearning.init_optimizer_cache
update!
```

## References

```@bibliography 
Pages = []
Canonical = false

goodfellow2016deep
```