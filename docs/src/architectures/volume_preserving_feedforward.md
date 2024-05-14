# Volume-Preserving Feedforward Neural Network 

## Neural network architecture

The constructor produces the following architecture[^1]:

[^1]: Based on the input arguments `n_linear` and `n_blocks`. In this example `init_upper` is set to false, which means that the first layer is of type *lower* followed by a layer of type *upper*. 

```@example
Main.include_graphics("../tikz/vp_feedforward") # hide
```

Here *LinearLowerLayer* performs ``x \mapsto x + Lx`` and *NonLinearLowerLayer* performs ``x \mapsto x + \sigma(Lx + b)``. The activation function ``\sigma`` is the forth input argument to the constructor and `tanh` by default. 

## Note on Sympnets

As [SympNets](@ref "SympNet Architecture") are symplectic maps, they also conserve phase space volume and therefore form a subcategory of volume-preserving feedforward layers. 

## Library Functions 

```@docs; canonical=false
VolumePreservingFeedForward
```