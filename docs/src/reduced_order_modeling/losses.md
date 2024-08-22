# Losses and Errors

In general we distinguish between *losses* that are used during training of a neural network and *errors* that arise in the context of reduced order modeling. 

## Different Neural Network Losses

`GeometricMachineLearning` has a number of loss functions implemented that can be called *standard losses*. Those are the [`FeedForwardLoss`](@ref), the [`TransformerLoss`](@ref), the [`AutoEncoderLoss`](@ref) and the [`ReducedLoss`](@ref). How to implement custom losses is shown in a [tutorial](@ref "Adjusting the Loss Function").

## A Note on Physics-Informed Neural Networks

A popular trend in recent years has been considering known physical properties of the differential equation, or the entire differential equation, through the loss function [raissi2019physics](@cite). This is one way of considering physical properties, and `GeometricMachineLearning` allows for a [flexible implementation of custom losses](@ref "Adjusting the Loss Function"), but this is nonetheless discouraged. In general a neural networks consists of *three ingredients*:

```@example
Main.include_graphics("../tikz/ingredients"; width = .98) # hide
```

Instead of considering certain properties through the loss function, we instead do so by enforcing them strongly through the network architecture and the optimizer; the latter pertains to [manifold optimization](@ref "Generalization to Homogeneous Spaces"). The advantages of this approach are the strong enforcement of properties that we know our network should have and much easier training because we do not have to tune hyperparameters. 


## Projection and Reduction Errors of Reduced Models

Two errors that are of very big importance in reduced order modeling are the *projection* and the *reduction error*. During training one typically aims at minimizing the projection error, but for the actual application of the model the reduction error is often more important.

## Projection Error 

The projection error computes how well a reduced basis, represented by the reduction ``\mathcal{P}`` and the reconstruction ``\mathcal{R}``, can represent the data with which it is build. In mathematical terms: 

```math
    e_\mathrm{proj}(\mu) := \frac{|| \mathcal{R}\circ\mathcal{P}(M) - M ||}{|| M ||},
```
where ``||\cdot||`` is the Frobenius norm (one could also optimize for different norms). The corresponding function in `GeometricMachineLearning` is [`projection_error`](@ref). The projection error is equivalent to [`AutoEncoderLoss`](@ref) and is used for training under that name.

## Reduction Error

The reduction error measures how far the reduced system diverges from the full-order system during integration (online stage). In mathematical terms (and for a single initial condition): 

```math
e_\mathrm{red}(\mu) := \sqrt{
    \frac{\sum_{t=0}^K|| \mathbf{x}^{(t)}(\mu) - \mathcal{R}(\mathbf{x}^{(t)}_r(\mu)) ||^2}{\sum_{t=0}^K|| \mathbf{x}^{(t)}(\mu) ||^2}
},
```
where ``\mathbf{x}^{(t)}`` is the solution of the FOM at point ``t`` and ``\mathbf{x}^{(t)}_r`` is the solution of the ROM (in the reduced basis) at point ``t``. The reduction error, as opposed to the projection error, not only measures how well the solution manifold is represented by the reduced basis, but also measures how well the FOM dynamics are approximated by the ROM dynamics (via the induced vector field on the reduced basis). The corresponding function in `GeometricMachineLearning` is [`reduction_error`](@ref). The reduction error is, in contract to the projection error, typically not used during training (even though some authors are using a similar error to do so [lee2020model](@cite)).

## Library Functions

```@docs
GeometricMachineLearning.NetworkLoss
FeedForwardLoss
TransformerLoss
AutoEncoderLoss
GeometricMachineLearning.ClassificationTransformerLoss
ReducedLoss
projection_error
reduction_error
```

## References

```@bibliography
Canonical = false
Pages = []

lee2020model
raissi2019physics
```