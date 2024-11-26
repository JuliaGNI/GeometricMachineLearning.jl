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
- `depth::Int = """ * "$(la_depth_default)`" * raw""": The number of linear layers that are applied. 
- `nhidden::Int = """ * "$(la_nhidden_default)`" * raw""": The number of hidden layers (i.e. layers that are *not* output layers).
- `activation = """ * "$(la_activation_default)`" * raw""": The activation function that is applied.
- `init_upper_linear::Bool = """ * "$(la_init_upper_linear_default)`" * raw""": Initialize the linear layer so that it first modifies the ``q``-component. 
- `init_upper_act::Bool = """ * "$(la_init_upper_act_default)`" * raw""": Initialize the activation layer so that it first modifies the ``q``-component.
"""
struct LASympNet{AT, InitUpperLinear, InitUpperAct} <: SympNet{AT} where {InitUpperLinear, InitUpperAct}
    dim::Int
    depth::Int
    nhidden::Int
    activation::AT

    function LASympNet(dim::Int;    depth = la_depth_default, 
                                    nhidden = la_nhidden_default, 
                                    activation = la_activation_default, 
                                    init_upper_linear = la_init_upper_linear_default, 
                                    init_upper_act = la_init_upper_act_default) 
        new{typeof(activation), init_upper_linear, init_upper_act}(dim, min(depth,5), nhidden, activation)
    end

    function LASympNet(dl::DataLoader;  depth = la_depth_default, 
                                        nhidden = la_nhidden_default, 
                                        activation = la_activation_default, 
                                        init_upper_linear = la_init_upper_linear_default, 
                                        init_upper_act = la_init_upper_act_default)
        new{typeof(activation), init_upper_linear, init_upper_act}(dl.input_dim, min(depth,5), nhidden, activation)
    end
end

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
- `n_layers::Int""" * "$(g_n_layers_default)`" * raw""": The number of layers (i.e. the total number of [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref)).
- `activation""" * "$(g_activation_default)`" * raw""": The activation function that is applied.
- `init_upper::Bool""" * "$(g_init_upper_default)`" * raw""": Initialize the gradient layer so that it first modifies the $q$-component.
"""
struct GSympNet{AT} <: SympNet{AT}
    dim::Int
    upscaling_dimension::Int
    n_layers::Int
    act::AT
    init_upper::Bool

    function GSympNet(dim;  upscaling_dimension = 2 * dim, 
                            n_layers = g_n_layers_default, 
                            activation = g_activation_default, 
                            init_upper = g_init_upper_default) 
        new{typeof(activation)}(dim, upscaling_dimension, n_layers, activation, init_upper)
    end

        
    function GSympNet(dl::DataLoader;   upscaling_dimension = 2 * dl.input_dim, 
                                        n_layers = g_n_layers_default, 
                                        activation = g_activation_default, 
                                        init_upper = g_init_upper_default) 
        new{typeof(activation)}(dl.input_dim, upscaling_dimension, n_layers, activation, init_upper)
    end
end

function Chain(arch::GSympNet)
    layers = ()
    is_upper_criterion = arch.init_upper ? isodd : iseven
    for i in 1:arch.n_layers
        layers = (layers..., 
        if is_upper_criterion(i)
            GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.act)
        else 
            GradientLayerP(arch.dim, arch.upscaling_dimension, arch.act)
        end
        )
    end
    Chain(layers...)
end

function Chain(arch::LASympNet{AT, true, false}) where {AT}
    layers = ()
    for _ in 1:arch.nhidden
        for j in 1:(arch.depth)
            layers = isodd(j) ? (layers..., LinearLayerQ(arch.dim)) : (layers..., LinearLayerP(arch.dim))
        end
        layers = (layers..., BiasLayer(arch.dim))
        layers = (layers..., ActivationLayerP(arch.dim, arch.activation))
        layers = (layers..., ActivationLayerQ(arch.dim, arch.activation))
    end
    for j in 1:(arch.depth)
        layers = isodd(j) ? (layers..., LinearLayerQ(arch.dim)) : (layers..., LinearLayerP(arch.dim))
    end
    layers = (layers..., BiasLayer(arch.dim))
    Chain(layers...)
end

function Chain(arch::LASympNet{AT, false, true}) where {AT}
    layers = ()
    for i in 1:arch.nhidden
        for j in 1:arch.depth
            layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
        end
        layers = (layers..., BiasLayer(arch.dim))
        layers = (layers..., ActivationLayerQ(arch.dim, arch.activation))
        layers = (layers..., ActivationLayerP(arch.dim, arch.activation))
    end
    for j in 1:(arch.depth)
        layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
    end
    layers = (layers..., BiasLayer(arch.dim))
    Chain(layers...)
end

function Chain(arch::LASympNet{AT, false, false}) where {AT}
    layers = ()
    for i in 1:arch.nhidden
        for j in 1:arch.depth
            layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
        end
        layers = (layers..., BiasLayer(arch.dim))
        layers = (layers..., ActivationLayerP(arch.dim, arch.activation))
        layers = (layers..., ActivationLayerQ(arch.dim, arch.activation))
    end
    for j in 1:(arch.depth)
        layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
    end
    layers = (layers..., BiasLayer(arch.dim))
    Chain(layers...)
end

function Chain(arch::LASympNet{AT, true, true}) where {AT}
    layers = ()
    for i in 1:arch.nhidden
        for j in 1:arch.depth
            layers = isodd(j) ? (layers..., LinearLayerQ(arch.dim)) : (layers..., LinearLayerP(arch.dim))
        end
        layers = (layers..., BiasLayer(arch.dim))
        layers = (layers..., ActivationLayerQ(arch.dim, arch.activation))
        layers = (layers..., ActivationLayerP(arch.dim, arch.activation))
    end
    for j in 1:(arch.depth)
        layers = isodd(j) ? (layers..., LinearLayerQ(arch.dim)) : (layers..., LinearLayerP(arch.dim))
    end
    layers = (layers..., BiasLayer(arch.dim))
    Chain(layers...)
end