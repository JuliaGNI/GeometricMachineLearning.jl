const prn_n_blocks_default = 1
const prn_activation_default = tanh

@doc raw"""
    ParametricResNet(dim; width, n_blocks, activation, parameters)

A [`ResNet`](@ref) whose blocks also take the parameters of the *system*, built from
[`ParametricResNetLayer`](@ref)s.

The architecture without structure preservation that the generalized Hamiltonian neural network of
[HamiltonianNeuralNetworks.jl](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/latest/) is
compared against: it maps ``(x, \mu) \mapsto x'`` with the same information available to it, but
nothing in it makes the map symplectic. Unlike `ResNet` it uses a `ParametricResNetLayer` at every
width, and the linear output layer applies the identity activation.

# Keyword arguments

- `width = dim`: the width of the hidden layer of each block.
- `n_blocks = """ * "$(prn_n_blocks_default)`" *
     raw""": the number of blocks before the linear one.
- `activation = """ * "$(prn_activation_default)`" *
     raw""": the activation of every block.
- `parameters = NullParameters()`: a `NamedTuple` of system parameters, used only for its shape.

# Constructor

`ParametricResNet` can also be called with the constructor

```julia
ParametricResNet(dl, n_blocks, width)
```

where `dl` is an instance of [`DataLoader`](@ref) and `width` defaults to `dl.input_dim`, or an
instance of [`ParametricDataLoader`](@ref). The latter gives the system dimension, the rows of its
data or of `q` and `p` together, which is also the default of `width`, and the system parameters,
whose first `μ` the architecture keeps for its shape; it therefore takes no `parameters` keyword.

The forward pass is [`apply_parametric`](@ref), or the functor `nn(x, μ)` of the `NeuralNetwork`
built around it.
"""
struct ParametricResNet{AT, PT} <: NeuralNetworkIntegrator
    sys_dim::Int
    n_blocks::Int
    width::Int
    parameters::PT
    activation::AT

    function ParametricResNet(dim; width = dim, n_blocks = prn_n_blocks_default,
            activation = prn_activation_default, parameters = NullParameters())
        new{typeof(activation), typeof(parameters)}(
            dim, n_blocks, width, parameters, activation)
    end
end

function ParametricResNet(dl::DataLoader, n_blocks::Integer, width::Integer = dl.input_dim;
        activation = prn_activation_default, parameters = NullParameters())
    ParametricResNet(dl.input_dim; width = width, n_blocks = n_blocks,
        activation = activation, parameters = parameters)
end

function ParametricResNet(dl::ParametricDataLoader, n_blocks::Integer,
        width::Integer = _system_dimension(dl.input); activation = prn_activation_default)
    ParametricResNet(_system_dimension(dl.input); width = width, n_blocks = n_blocks,
        activation = activation, parameters = first(dl.parameters))
end

# The rows of a `2d × nt × m` tensor, or of the `q` and `p` of a `(q, p)` pair of them.
_system_dimension(input::AbstractArray{<:Any, 3}) = size(input, 1)
_system_dimension(input::NamedTuple{(:q, :p)}) = size(input.q, 1) + size(input.p, 1)

function Chain(arch::ParametricResNet{AT}) where {AT}
    layers = ()
    for _ in 1:arch.n_blocks
        # nonlinear layers
        layers = (layers...,
            ParametricResNetLayer(arch.sys_dim, arch.width, arch.activation;
                parameters = arch.parameters, return_parameters = true))
    end

    # linear layer for the output
    layers = (layers...,
        ParametricResNetLayer(arch.sys_dim, arch.width, identity;
            parameters = arch.parameters, return_parameters = false))

    Chain(layers...)
end

# Without this, `nn(x, μ)` reaches `AbstractNeuralNetworks`' generic two-argument functor, which
# reads the system parameters as the *network* parameters and hands the first layer a `Float64`.
# A `ParametricResNet` is a `NeuralNetworkIntegrator`, which carries no method of its own for a
# network whose forward pass takes the parameters of the system. Every input goes to
# `apply_parametric`, one `μ` or a vector of them, which raises for a shape it does not take.
function (nn::NeuralNetwork{<:ParametricResNet})(
        x, μ::Union{OptionalParameters, AbstractVector{<:NamedTuple}})
    apply_parametric(nn.model, x, μ, params(nn))
end
