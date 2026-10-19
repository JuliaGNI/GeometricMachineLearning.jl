@doc raw"""
    SympNet <: NeuralNetworkIntegrator

The `SympNet` type encompasses [`GSympNet`](@ref)s and [`LASympNet`](@ref)s. 
SympNets [jin2020sympnets](@cite) are universal approximators of *canonical symplectic flows*.
This means that for every map 
```math
    \varphi:\mathbb{R}^{2n}\to\mathbb{R}^{2n} \text{ with } (\nabla\varphi)^T\mathbb{J}\nabla\varphi = \mathbb{J},
``` 
we can find a SympNet that approximates ``\varphi`` arbitrarily well.
"""
abstract type SympNet{AT} <: NeuralNetworkIntegrator end

const la_depth_default = 5
const la_nhidden_default = 1
const la_init_upper_linear_default = true
const la_init_upper_act_default = true
const la_activation_default = tanh

@doc raw"""
    LASympNet(d)

Make an ``LA``-SympNet with dimension ``d.``

There exists an additional constructor that can be called by supplying an instance of [`DataLoader`](@ref).

# Examples

```jldoctest
using GeometricMachineLearning
dl = DataLoader(rand(2, 20); suppress_info = true)
LASympNet(dl)

# output

LASympNet{typeof(tanh), true, true}(2, 5, 1, tanh)
```

# Arguments

Keyword arguments are: 
- `depth::Int = """ * "$(la_depth_default)`" *
     raw""": The number of linear layers that are applied. 
- `nhidden::Int = """ * "$(la_nhidden_default)`" *
     raw""": The number of hidden layers (i.e. layers that are *not* output layers).
- `activation = """ * "$(la_activation_default)`" *
     raw""": The activation function that is applied.
- `init_upper_linear::Bool = """ * "$(la_init_upper_linear_default)`" *
     raw""": Initialize the linear layer so that it first modifies the ``q``-component. 
- `init_upper_act::Bool = """ * "$(la_init_upper_act_default)`" *
     raw""": Initialize the activation layer so that it first modifies the ``q``-component.
"""
struct LASympNet{AT, InitUpperLinear, InitUpperAct} <:
       SympNet{AT} where {InitUpperLinear, InitUpperAct}
    dim::Int
    depth::Int
    nhidden::Int
    activation::AT
end

function LASympNet(dim::Int; depth = la_depth_default,
        nhidden = la_nhidden_default,
        activation = la_activation_default,
        init_upper_linear::Bool = la_init_upper_linear_default,
        init_upper_act::Bool = la_init_upper_act_default)
    LASympNet{typeof(activation), init_upper_linear, init_upper_act}(
        dim, min(depth, 5), nhidden, activation)
end

# A `DataLoader` names the dimension and nothing else, so its constructor is the dimension one.
LASympNet(dl::DataLoader; kwargs...) = LASympNet(dl.input_dim; kwargs...)

@inline AbstractNeuralNetworks.dim(arch::SympNet) = arch.dim

const g_n_layers_default = 2
const g_activation_default = tanh
const g_init_upper_default = true

@doc raw"""
    GSympNet(d)

Make a ``G``-SympNet with dimension ``d.``

There exists an additional constructor that can be called by supplying an instance of [`DataLoader`](@ref) (see [`LASympNet`](@ref) for an example of using this constructor).

# Arguments

Keyword arguments are:
- `upscaling_dimension::Int = 2d`: The *upscaling dimension* of the gradient layer. See the documentation for [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref) for further explanation.
- `n_layers::Int""" * "$(g_n_layers_default)`" *
     raw""": The number of layers (i.e. the total number of [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref)).
- `activation""" * "$(g_activation_default)`" *
     raw""": The activation function that is applied.
- `init_upper::Bool""" * "$(g_init_upper_default)`" *
     raw""": Initialize the gradient layer so that it first modifies the $q$-component.
"""
struct GSympNet{AT} <: SympNet{AT}
    dim::Int
    upscaling_dimension::Int
    n_layers::Int
    activation::AT
    init_upper::Bool
end

function GSympNet(dim; upscaling_dimension = 2 * dim,
        n_layers = g_n_layers_default,
        activation = g_activation_default,
        init_upper::Bool = g_init_upper_default)
    GSympNet{typeof(activation)}(dim, upscaling_dimension, n_layers, activation, init_upper)
end

# A `DataLoader` names the dimension and the upscaling, so its constructor is the dimension one.
function GSympNet(dl::DataLoader; upscaling_dimension = 2 * dl.input_dim, kwargs...)
    GSympNet(dl.input_dim; upscaling_dimension, kwargs...)
end

# `n` gradient layers, alternating between the two halves, the first of them acting on `q` when
# `init_upper`. Both the SympNet and the transformer architectures build their blocks with this.
function _gradient_layers(dim::Integer, upscaling_dimension::Integer, activation,
        n::Integer, init_upper::Bool)
    is_upper_criterion = init_upper ? isodd : iseven
    layers = ()
    for i in 1:n
        layers = is_upper_criterion(i) ?
                 (layers..., GradientLayerQ(dim, upscaling_dimension, activation)) :
                 (layers..., GradientLayerP(dim, upscaling_dimension, activation))
    end
    layers
end

function Chain(arch::GSympNet)
    Chain(_gradient_layers(arch.dim, arch.upscaling_dimension, arch.activation,
        arch.n_layers, arch.init_upper)...)
end

# One chain for all four `(init_upper_linear, init_upper_act)` pairs: the linear layers start with
# the half `InitUpperLinear` names and alternate, and the two activation layers are `(q, p)` in that
# order when `InitUpperAct` is set and `(p, q)` otherwise.
function Chain(arch::LASympNet{
        AT, InitUpperLinear, InitUpperAct}) where {AT, InitUpperLinear,
        InitUpperAct}
    is_upper_criterion = InitUpperLinear ? isodd : iseven
    layers = ()
    for _ in 1:arch.nhidden
        for j in 1:arch.depth
            layers = is_upper_criterion(j) ? (layers..., LinearLayerQ(arch.dim)) :
                     (layers..., LinearLayerP(arch.dim))
        end
        layers = (layers..., BiasLayer(arch.dim))
        layers = InitUpperAct ?
                 (layers..., ActivationLayerQ(arch.dim, arch.activation),
            ActivationLayerP(arch.dim, arch.activation)) :
                 (layers..., ActivationLayerP(arch.dim, arch.activation),
            ActivationLayerQ(arch.dim, arch.activation))
    end
    for j in 1:arch.depth
        layers = is_upper_criterion(j) ? (layers..., LinearLayerQ(arch.dim)) :
                 (layers..., LinearLayerP(arch.dim))
    end
    layers = (layers..., BiasLayer(arch.dim))
    Chain(layers...)
end
