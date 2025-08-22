@doc raw"""
    ResNet(dim, n_blocks, activation)

Make an instance of a `ResNet`.

A ResNet is a neural network that realizes a mapping of the form: 

```math
    x = \mathcal{NN}(x) + x,
```
so the input is again added to the output (a so-called add connection). 
In `GeometricMachineLearning` the specific ResNet that we use consists of a series of simple [`ResNetLayer`](@ref)s.

# Constructor

`ResNet` can also be called with the constructor:
```julia
ResNet(dl, n_blocks)
```

where `dl` is an instance of `DataLoader`.

See [`iterate`](@ref) for an example of this.
"""
struct ResNet{AT} <: NeuralNetworkIntegrator
    sys_dim::Int
    n_blocks::Int
    width::Int
    activation::AT
end

function ResNet(dl::DataLoader, n_blocks::Integer, width::Integer=dl.input_dim; activation = tanh)
    ResNet(dl.input_dim, n_blocks, width, activation)
end

function Chain(arch::ResNet{AT}) where AT
    layers = ()
    for _ in 1:arch.n_blocks 
        # nonlinear layers
        layers = (layers..., arch.sys_dim == arch.width ? ResNetLayer(arch.sys_dim, arch.activation; use_bias=true) : WideResNetLayer(arch.sys_dim, arch.width, arch.activation))
    end

    # linear layers for the output
    layers = (layers..., ResNetLayer(arch.sys_dim, identity; use_bias=true))

    Chain(layers...)
end