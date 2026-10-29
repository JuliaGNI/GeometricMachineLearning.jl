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

There exists an additional constructor that can be called by supplying an instance of `DataLoader`.

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

    function LASympNet(dim::Int; depth = la_depth_default,
            nhidden = la_nhidden_default,
            activation = la_activation_default,
            init_upper_linear = la_init_upper_linear_default,
            init_upper_act = la_init_upper_act_default)
        new{typeof(activation), init_upper_linear, init_upper_act}(dim, min(depth, 5), nhidden, activation)
    end
end

LASympNet(dl::DataLoader; kwargs...) = LASympNet(dl.input_dim; kwargs...)

@inline AbstractNeuralNetworks.dim(arch::SympNet) = arch.dim

const g_n_layers_default = 2
const g_activation_default = tanh
const g_init_upper_default = true

@doc raw"""
    GSympNet(d)

Make a ``G``-SympNet with dimension ``d.``

There exists an additional constructor that can be called by supplying an instance of `DataLoader` (see [`LASympNet`](@ref) for an example of using this constructor).

# Arguments

Keyword arguments are:
- `upscaling_dimension::Int = 2d`: The *upscaling dimension* of the gradient layer. See the documentation for [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref) for further explanation.
- `n_layers::Int = """ * "$(g_n_layers_default)`" *
     raw""": The number of layers (i.e. the total number of [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref)).
- `activation = """ * "$(g_activation_default)`" *
     raw""": The activation function that is applied.
- `init_upper::Bool = """ * "$(g_init_upper_default)`" *
     raw""": Initialize the gradient layer so that it first modifies the $q$-component.
"""
struct GSympNet{AT} <: SympNet{AT}
    dim::Int
    upscaling_dimension::Int
    n_layers::Int
    activation::AT
    init_upper::Bool

    function GSympNet(dim; upscaling_dimension = 2 * dim,
            n_layers = g_n_layers_default,
            activation = g_activation_default,
            init_upper = g_init_upper_default)
        new{typeof(activation)}(dim, upscaling_dimension, n_layers, activation, init_upper)
    end
end

GSympNet(dl::DataLoader; kwargs...) = GSympNet(dl.input_dim; kwargs...)

# `n` gradient layers that alternate between `GradientLayerQ` and `GradientLayerP`, starting with
# `GradientLayerQ` if `init_upper` holds.
function _gradient_layers(
        dim::Integer, upscaling_dimension::Integer, activation, n::Integer,
        init_upper::Bool)
    ntuple(
        i -> isodd(i) == init_upper ? GradientLayerQ(dim, upscaling_dimension, activation) :
             GradientLayerP(dim, upscaling_dimension, activation),
        n)
end

function Chain(arch::GSympNet)
    Chain(_gradient_layers(
        arch.dim, arch.upscaling_dimension, arch.activation, arch.n_layers, arch.init_upper)...)
end

# `nhidden` blocks of `depth` linear layers, a bias and two activation layers, then `depth` linear
# layers and a bias. The linear layers alternate and start with `LinearLayerQ` if
# `InitUpperLinear`; the activation layers are `ActivationLayerQ` first if `InitUpperAct`.
function Chain(arch::LASympNet{
        <:Any, InitUpperLinear, InitUpperAct}) where {InitUpperLinear, InitUpperAct}
    linear_layers = ntuple(
        j -> isodd(j) == InitUpperLinear ? LinearLayerQ(arch.dim) : LinearLayerP(arch.dim),
        arch.depth)
    q_activation = ActivationLayerQ(arch.dim, arch.activation)
    p_activation = ActivationLayerP(arch.dim, arch.activation)
    activation_layers = InitUpperAct ? (q_activation, p_activation) :
                        (p_activation, q_activation)
    layers = ()
    for _ in 1:arch.nhidden
        layers = (layers..., linear_layers..., BiasLayer(arch.dim), activation_layers...)
    end
    Chain(layers..., linear_layers..., BiasLayer(arch.dim))
end
