# Standard Transformer

The transformer is a relatively modern neural network architecture [vaswani2017attention](@cite) that has come to dominate the field of natural language processing (NLP, [patwardhan2023transformers](@cite)) and replaced the previously dominant long-short term memory cells (LSTM, [hochreiter1997long](@cite)). Its success is due to a variety of factors: 
- unlike LSTMs it consists of very simple building blocks and hence is easier to interpret mathematically,
- it is very flexible in its application and the data it is fed with do not have to conform to a rigid pattern, 
- transformers utilize modern hardware (especially GPUs) very effectively. 

The transformer architecture is sketched below: 

```@example
Main.include_graphics("../tikz/transformer_encoder") # hide
```

It is nothing more than a combination of a [multihead attention layer](@ref "Multihead Attention") and a residual neural network[^1] (ResNet).

[^1]: A ResNet is nothing more than a neural network to whose output we again add the input, i.e. every ResNet is of the form ``\mathrm{ResNet}(x) = x + \mathcal{NN}(x)``.

As was explained when we talked about the [attention module](@ref "Reweighting of the Input Sequence"), the attention layer performs a convex reweighting of the input sequence:

```math
\mathrm{Attention}:  Z \equiv [z^{(1)}, \ldots, z^{(T)}]   \mapsto  [\sum_{i=1}^Tp^{(1)}_iz^{(i)}, \ldots, \sum_{i=1}^Tp^{(T)}_iz^{(i)}] = \mathrm{Attention}(Z),
```
where the coefficients ``p^{(i)}`` depend on ``Z`` and are [*learnable*](@ref "The Attention Layer"). In the case of [multihead attention](@ref "Multihead Attention") a greater number of these *reweighting coefficients* are learned, but it is otherwise not much more complicated than single-head attention.

The green arrow in the figure above indicates that this first *add connection* can be left out. This can be specified via the keyword argument `add_connection` in [`StandardTransformerIntegrator`](@ref) and the [`MultiHeadAttention`](@ref) layer.

We should also note that such transformers have been used for [the online phase in reduced order modeling](@ref "General Workflow") before [solera2023beta](@cite).

## Classification Transformer

Instead of using the transformer for integration, it can also be used as a image classifier. In this case it is often referred to as "vision transformer" [dosovitskiy2020image](@cite). In this case we append a [`ClassificationLayer`](@ref) to the output of the transformer. This will be used in the [MNIST tutorial](@ref "MNIST Tutorial"). 

## The Upscaling

When using the transformer one typically also benefits from defining a `transformer_dim` that is greater than the system dimension and a corresponding `upscaling_activation` (see the docstring of [`StandardTransformerIntegrator`](@ref)).

```@example
Main.include_graphics("../tikz/transformer_upscaling"; caption = raw"If the transformer dimension is not equal to the system dimension, then we add two more neural network layers. One that maps from the system dimension to the transformer dimension and one that maps from the transformer dimension to the system dimension.") # hide
```

In the figure above we call 

```math
    \Psi^\mathrm{up}:\mathbb{R}^{\mathtt{sys\_dim}}\to\mathbb{R}^{\mathtt{transformer\_dim}}
```

the *upscaling layer* and 

```math
    \Psi^\mathrm{down}:\mathbb{R}^{\mathtt{transformer\_dim}}\to\mathbb{R}^{\mathtt{sys\_dim}}
```
the *downscaling layer*. Both of these layers are dense layers with the activation function for the downscaling layer being the identity (for better expressivity) and the activation function for the upscaling layer can be specified via the keyword `upscaling_activation` (it is `tanh` by default).

At the moment `GeometricMachineLearning` does not have an implementation of such an upscaling for the [volume-preserving transformer](@ref "Volume-Preserving Transformer") and the [linear symplectic transformer](@ref "Linear Symplectic Transformer"). Symplectic liftings have however recently been discussed to learn higher-dimensional Hamiltonian representations of given data [yildiz2024data](@cite) and this approach could also be adapted here. This is left for future work.

## Library Functions 

```@docs
StandardTransformerIntegrator
Transformer
ClassificationTransformer
GeometricMachineLearning.assign_output_estimate
```

## References

```@bibliography
Pages = []
Canonical = false

vaswani2017attention
solera2023beta
```