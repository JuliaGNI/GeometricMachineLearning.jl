@doc raw"""
    ParametricResNet(dim; width, n_blocks, activation, parameters)

A [`ResNet`](@ref) whose blocks also take the parameters of the *system*, built from
[`ParametricResNetLayer`](@ref)s.

This is the architecture without structure preservation that
[`GeneralizedHamiltonianArchitecture`](@ref) is compared against: it maps
``(x, \mu) \mapsto x'`` with the same information available to it, but nothing in it makes the map
symplectic.

# Keyword arguments

- `width = dim`: the width of the hidden layer of each block,
- `n_blocks = $(HNN_nhidden_default)`: the number of blocks,
- `activation = $(HNN_activation_default)`,
- `parameters = NullParameters()`: a `NamedTuple` of system parameters, used for its shape.

`ResNet(dim; n_blocks, width, parameters)` dispatches here when `parameters` is anything other than
`NullParameters`.
"""
struct ParametricResNet{AT <: Activation, PT <: OptionalParameters} <: NeuralNetworkIntegrator
    sys_dim::Int
    n_blocks::Int
    width::Int
    parameters::PT
    activation::AT

    function ParametricResNet(dim; width=dim, n_blocks = HNN_nhidden_default, activation=HNN_activation_default, parameters=NullParameters())
        activation = (typeof(activation) <: Activation) ? activation : Activation(activation)
        new{typeof(activation), typeof(parameters)}(dim, n_blocks, width, parameters, activation)
    end
end

function ParametricResNet(dl::DataLoader, n_blocks::Integer, width::Integer=dl.input_dim; activation=HNN_activation_default, parameters=NullParameters())
    ParametricResNet(dl.input_dim; width=width, n_blocks=n_blocks, activation=activation, parameters=parameters)
end

function ResNet(input_dim::Integer, n_blocks::Integer, width::Integer=input_dim; activation=HNN_activation_default, parameters=NullParameters())
    typeof(parameters) <: NullParameters ? ResNet(input_dim, n_blocks, width, activation) : ParametricResNet(input_dim; n_blocks=n_blocks, width=width, parameters=parameters, activation=activation)
end

function ResNet(input_dim::Integer; n_blocks::Integer, width::Integer=input_dim, activation=HNN_activation_default, parameters=NullParameters())
    ResNet(input_dim, n_blocks, width; activation=activation, parameters=parameters)
end

function Chain(arch::ParametricResNet{AT}) where AT
    layers = ()
    for _ in 1:arch.n_blocks 
        # nonlinear layers
        layers = (layers..., ParametricResNetLayer(arch.sys_dim, arch.width, arch.activation; parameters=arch.parameters, return_parameters=true))
    end

    # linear layers for the output
    layers = (layers..., ParametricResNetLayer(arch.sys_dim, arch.width, identity; parameters=arch.parameters, return_parameters=false))

    Chain(layers...)
end