# Different Neural Network Losses

`GeometricMachineLearning` has a number of loss functions implemented that can be called *standard losses*. How to implement custom losses is shown in the [tutorials](@ref "Adjusting the Loss Function").

## A Note on Physics-Informed Neural Networks

A popular trend in recent years has been considering known physical properties of the differential equation, or the entire differential equation, through the loss function [raissi2019physics](@cite). This is one way of considering physical properties, and `GeometricMachineLearning` allows for a [flexible implementation of custom losses](@ref "Adjusting the Loss Function"), but this is nonetheless discouraged. In general a neural networks consists of *three ingredients*:

```@example
Main.include_graphics("../tikz/ingredients") # hide
```

Instead of considering certain properties through the loss function, we instead do so by enforcing them strongly through the network architecture and the optimizer; the latter pertains to [manifold optimization](@ref "Generalization to Homogeneous Spaces"). The advantages of this approach are the strong enforcement of properties that we now our network should have and much easier training because we do not have to tune hyperparameters. 


```@docs; canonical = false
GeometricMachineLearning.NetworkLoss
TransformerLoss
FeedForwardLoss
AutoEncoderLoss
ReducedLoss
```

```@bibliography
Canonical = false
Pages = []

raissi2019physics
```