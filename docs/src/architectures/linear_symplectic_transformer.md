# Linear Symplectic Transformer

The linear symplectic transformer consists of a combination of [linear symplectic attention](@ref "Linear Symplectic Attention") and [gradient layers](@ref "SympNet Gradient Layer") and is visualized below:

```@example
Main.include_graphics("../tikz/linear_symplectic_transformer"; caption = raw"Visualization of the linear symplectic transformer architecutre. In this figure the number of SympNet layers per transformer block is two. ", width = .3) # hide
```

In this picture we also visualize the keywords `n_sympnet` and ``L`` for [`LinearSymplecticTransformer`](@ref).

What we discussed for the [volume-preserving transformer](@ref "Volume-Preserving Transformer") also applies here: the attention mechanism acts on all the input vectors at once and is designed such that it preserves the product structure (here this is the symplectic product structure). The attention mechanism serves as a *preprocessing step* after which we apply a regular feedforward neural network; here this is a [SympNet](@ref "SympNet Architecture").

## Why use Transformers for Model Order Reduction

The [standard transformer](@ref "Standard Transformer"), the [volume-preserving transformer](@ref "Volume-Preserving Transformer") and the linear symplectic transformer are suitable for model order reduction for a number of reasons. Besides their improved accuracy [solera2023beta](@cite) their ability to resolve time series data also makes it possible to deal with data that come from multiple parameters. For this consider the following two trajectories:


```@example
Main.include_graphics("../tikz/multiple_parameters"; caption = raw"Two trajectories of a parameter-dependent ODE with the same initial condition. ") # hide
```

The trajectories come from a parameter-dependent [ODE](@ref "The Existence-And-Uniqueness Theorem") in two dimensions. As initial condition we take ``A\in\mathbb{R}^2`` and we look at two different parameter instances: ``\mu_1`` and ``\mu_2``. As we can see the curves ``\tilde{z}_{\mu_1}`` and ``\tilde{z}_{\mu_2}`` both start out at ``A,`` then go into different directions but cross again at ``D.`` If we used a standard feedforward neural network to treat this system it would not be able to resolve those training data as the information would be ambiguous at points ``A`` and ``D,`` i.e. the network would not know what it should predict. If we however consider the information coming from points three points, either ``(A, B, D)`` or ``(A, C, D),`` then the network can learn to predict the next time step. We will elaborate more on this in the [tutorial section](@ref "Comparing Different `VolumePreservingAttention` Mechanisms").

## Library Functions

```@docs
LinearSymplecticTransformer
```