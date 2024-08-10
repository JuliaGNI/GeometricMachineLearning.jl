@doc raw"""
    ResNet(dim, n_blocks, activation)

Make an instance of a `ResNet`.

A ResNet is a neural network that realizes a mapping of the form: 

```math
    x = \mathcal{NN}(x) + x,
```
so the input is again added to the output (a so-called add connection). 
In `GeometricMachineLearning` the specific ResNet that we use consists of a series of simple [`ResNetLayer`](@ref)s.
"""
struct ResNet{AT} <: NeuralNetworkIntegrator
    sys_dim::Int 
    n_blocks::Int 
    activation::AT
end

function Chain(arch::ResNet{AT}) where AT
    layers = ()
    for _ in 1:arch.n_blocks 
        # nonlinear layers
        layers = (layers..., ResNetLayer(arch.sys_dim, arch.activation; use_bias=true))
    end

    # linear layers for the output
    layers = (layers..., ResNetLayer(arch.sys_dim, identity; use_bias=true))

    Chain(layers...)
end