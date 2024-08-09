# Volume-Preserving Feedforward Neural Network 

The volume-preserving feedforward neural network presented here can be seen as an adaption of an ``LA``-SympNet to the setting when we deal with a [divergence-free vector field](@ref "Divergence-Free Vector Fields"). It also serves as the feedforward module in the [volume-preserving transformer](@ref "Volume-Preserving Transformer"). 

## Neural network architecture

The constructor produces the following architecture[^1]:

[^1]: Based on the input arguments `n_linear` and `n_blocks`. In this example `init_upper` is set to false, which means that the first layer is of type *lower* followed by a layer of type *upper*. 

```@example
Main.include_graphics("../tikz/vp_feedforward") # hide
```

Here *LinearLowerLayer* performs ``x \mapsto x + Lx`` and *NonLinearLowerLayer* performs ``x \mapsto x + \sigma(Lx + b)``. The activation function ``\sigma`` is the forth input argument to the constructor and `tanh` by default. We can make an instance of a `VolumePreservingFeedForward` neural network:

```@example
using GeometricMachineLearning

const d = 3

arch = VolumePreservingFeedForward(d)

Chain(arch).layers
```

And we see that we get the same architecture as shown in the figure above, with the difference that the bias has been subsumed in the previous layers. Note that the nonlinear layers also contain a bias vector.

## Note on Sympnets

In the general framework of feedforward neural networks [SympNets](@ref "SympNet Architecture") are more restrictive than volume-preserving neural networks as symplecticity is a stronger property than volume-preservation:

```@example
Main.include_graphics("../tikz/structure_preservation_hierarchy"; caption = "Symplectic neural networks are a more restrictive class of architectures than volume-preserving ones. They also only work in even dimension.") # hide
```

Note however that SympNets rely on data in canonical form, i.e. data that is of [``(q, p)`` type](@ref "`GeometricMachineLearning.QPT`"), so those data need to come from a vector space ``\mathbb{R}^{2n}`` of even dimension. Volume-preserving feedforward neural networks also work for odd-dimensional spaces. This is also true for transformers: the [volume-preserving transformer](@ref "Volume-Preserving Transformer") works in spaces of arbitrary dimension, whereas the [linear symplectic transformer](@ref "Linear Symplectic Transformer") only works in even-dimensional spaces.

## Library Functions 

```@docs
VolumePreservingFeedForward
```