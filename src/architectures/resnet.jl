@doc raw"""
    ResNet(dim, n_blocks, activation)
    ResNet(dim, n_blocks, width, activation)

Make an instance of a `ResNet`.

A ResNet is a neural network that realizes a mapping of the form: 

```math
    x = \mathcal{NN}(x) + x,
```
so the input is again added to the output (a so-called add connection). 
In `GeometricMachineLearning` the specific ResNet that we use consists of a series of simple [`ResNetLayer`](@ref)s, or of [`WideResNetLayer`](@ref)s where `width` differs from `dim`: that is the one structural choice the architecture makes, and a `width` equal to `dim` keeps the plain `ResNetLayer`.

# Arguments

- `dim::Integer`: the system dimension.
- `n_blocks::Integer`: the number of blocks before the linear output layer.
- `width::Integer`: the width of the hidden layer of each block, which is `dim` in
  `ResNet(dim, n_blocks, activation)` and the third positional argument of
  `ResNet(dim, n_blocks, width, activation)` and of `ResNet(dl, n_blocks, width)`, where it
  defaults to `dl.input_dim`. See [`WideResNetLayer`](@ref).
- `activation`: the third positional argument of `ResNet(dim, n_blocks, activation)` and the fourth
  of `ResNet(dim, n_blocks, width, activation)`, where `ResNet(dl, n_blocks, width)` takes it as
  the keyword `activation = tanh`. It applies to every block but the last, which applies the
  identity.

# Constructor

`ResNet` can also be called with the constructor:
```julia
ResNet(dl, n_blocks, width)
```

where `dl` is an instance of `DataLoader` and `width` defaults to `dl.input_dim`.

See [`integrate`](@ref) for an example of this.

For a ResNet whose blocks also take the parameters of the system, see
[`ParametricResNet`](@ref).
"""
struct ResNet{AT} <: NeuralNetworkIntegrator
    sys_dim::Int
    n_blocks::Int
    width::Int
    activation::AT
end

function ResNet(sys_dim::Integer, n_blocks::Integer, activation)
    ResNet(sys_dim, n_blocks, sys_dim, activation)
end

function ResNet(dl::DataLoader, n_blocks::Integer, width::Integer = dl.input_dim;
        activation = tanh)
    ResNet(dl.input_dim, n_blocks, width, activation)
end

function Chain(arch::ResNet{AT}) where {AT}
    layers = ()
    for _ in 1:arch.n_blocks
        # nonlinear layers
        layers = (layers...,
            arch.sys_dim == arch.width ?
            ResNetLayer(arch.sys_dim, arch.activation; use_bias = true) :
            WideResNetLayer(arch.sys_dim, arch.width, arch.activation))
    end

    # linear layers for the output
    layers = (layers..., ResNetLayer(arch.sys_dim, identity; use_bias = true))

    Chain(layers...)
end
