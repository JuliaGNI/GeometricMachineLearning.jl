# Volume-Preserving Feedforward Neural Network 

The volume-preserving feedforward neural network presented here can be seen as an adaptation of an ``LA``-SympNet to the setting when we deal with a [divergence-free vector field](@ref "Divergence-Free Vector Fields"). It also serves as the feedforward module in the [volume-preserving transformer](@ref "Volume-Preserving Transformer"). 

## Neural network architecture

The constructor for [`VolumePreservingFeedForward`](@ref) produces the following architecture[^1]:

[^1]: Based on the input arguments `n_linear` and `n_blocks`. In this example `init_upper` is set to false, which means that the first layer is of type *lower* followed by a layer of type *upper*. 

```@example
Main.include_graphics("../tikz/vp_feedforward"; width = .25, caption = raw"Visualization of how the keywords in the constructor are interpreted. ") # hide
```

Here "LinearLowerLayer" performs 

```math
\mathrm{LinearLowerLayer}_{L}: x \mapsto x + Lx
``` 
and "NonLinearLowerLayer" performs 
```math
\mathrm{NonLinearLowerLayer}_{L}: x \mapsto x + \sigma(Lx + b). 
```

The activation function ``\sigma`` is the forth input argument to the constructor and `tanh` by default. We can make an instance of a [`VolumePreservingFeedForward`](@ref) neural network:

```@example
using GeometricMachineLearning

const d = 3

arch = VolumePreservingFeedForward(d)

@assert typeof(Chain(arch).layers[1]) <: VolumePreservingLowerLayer{3, 3, :no_bias, typeof(identity)} # hide
@assert typeof(Chain(arch).layers[2]) <: VolumePreservingUpperLayer{3, 3, :bias, typeof(identity)} # hide
@assert typeof(Chain(arch).layers[3]) <: VolumePreservingLowerLayer{3, 3, :bias, typeof(tanh)} # hide
@assert typeof(Chain(arch).layers[4]) <: VolumePreservingUpperLayer{3, 3, :bias, typeof(tanh)} # hide
@assert typeof(Chain(arch).layers[5]) <: VolumePreservingLowerLayer{3, 3, :no_bias, typeof(identity)} # hide
@assert typeof(Chain(arch).layers[6]) <: VolumePreservingUpperLayer{3, 3, :bias, typeof(identity)} # hide

for layer in Chain(arch)
    println(stdout, layer)
end
```

And we see that we get the same architecture as shown in the figure above, with the difference that the bias has been subsumed in the previous layers. Note that the nonlinear layers also contain a bias vector.

## Note on Sympnets

In the general framework of feedforward neural networks [SympNets](@ref "SympNet Architecture") are more restrictive than volume-preserving neural networks as symplecticity is a stronger property than volume-preservation:

```@example
Main.include_graphics("../tikz/structure_preservation_hierarchy"; width = .35, caption = "Symplectic neural networks are a more restrictive class of architectures than volume-preserving ones. But by construction they only work in even dimensions. ") # hide
```

Note however that SympNets rely on data in canonical form, i.e. data that is of ``(q, p)`` type (called [`GeometricMachineLearning.QPT`](@ref) in `GeometricMachineLearning`), so those data need to come from a vector space ``\mathbb{R}^{2n}`` of even dimension. Volume-preserving feedforward neural networks also work for odd-dimensional spaces. This is also true for transformers: the [volume-preserving transformer](@ref "Volume-Preserving Transformer") works in spaces of arbitrary dimension ``\mathbb{R}^{n\times{}T}``, whereas the [linear symplectic transformer](@ref "Linear Symplectic Transformer") only works in even-dimensional spaces ``\mathbb{R}^{2n\times{}T}``.

## Library Functions 

```@docs
VolumePreservingFeedForward
```