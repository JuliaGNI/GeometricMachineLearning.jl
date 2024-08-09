@doc raw"""
The `SympNet` type encompasses [`GSympNet`](@ref)s and [`LASympNet`](@ref)s. 
SympNets [jin2020sympnets](@cite) are universal approximators of *canonical symplectic flows*.
This means that for every map 
```math
    \varphi:\mathbb{R}^{2n}\to\mathbb{R}^{2n},
``` 
for which ``(\nabla\varphi)^T\mathbb{J}\nabla\varphi = \mathbb{J}`` holds, we can find a SympNet that approximates ``\varphi`` arbitrarily well.
"""
abstract type SympNet{AT} <: NeuralNetworkIntegrator end

@doc raw"""
    LASympNet(d)

Make an ``LA``-SympNet with dimension ``d.``

There exists an additional constructor that can be called by supplying an instance of [`DataLoader`](@ref).

# Arguments

Keyword arguments are: 
- `depth::Int`: The number of linear layers that are applied. The default is 5.
- `nhidden::Int`: The number of hidden layers (i.e. layers that are **not** output layers). The default is 2.
- `activation`: The activation function that is applied. By default this is `tanh`.
- `init_upper_linear::Bool`: Initialize the linear layer so that it first modifies the ``q``-component. The default is `true`.
- `init_upper_act::Bool`: Initialize the activation layer so that it first modifies the ``q``-component. The default is `true`.
"""
struct LASympNet{AT, InitUpperLinear, InitUpperAct} <: SympNet{AT} where {InitUpperLinear, InitUpperAct}
    dim::Int
    depth::Int
    nhidden::Int
    activation::AT

    function LASympNet(dim::Int; depth=5, nhidden=1, activation=tanh, init_upper_linear=true, init_upper_act=true) 
        new{typeof(activation), init_upper_linear, init_upper_act}(dim, min(depth,5), nhidden, activation)
    end

    function LASympNet(dl::DataLoader; depth=5, nhidden=1, activation=tanh, init_upper_linear=true, init_upper_act=true)
        new{typeof(activation), init_upper_linear, init_upper_act}(dl.input_dim, min(depth,5), nhidden, activation)
    end
end

@inline AbstractNeuralNetworks.dim(arch::SympNet) = arch.dim

@doc raw"""
    GSympNet(d)

Make a ``G``-SympNet with dimension ``d.``

There exists an additional constructor that can be called by supplying an instance of [`DataLoader`](@ref).

# Arguments

Keyword arguments are:
- `upscaling_dimension::Int`: The *upscaling dimension* of the gradient layer. See the documentation for `GradientLayerQ` and `GradientLayerP` for further explanation. The default is `2*dim`.
- `n_layers::Int`: The number of layers (i.e. the total number of [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref)). The default is 2.
- `activation`: The activation function that is applied. By default this is `tanh`.
- `init_upper::Bool`: Initialize the gradient layer so that it first modifies the $q$-component. The default is `true`.
"""
struct GSympNet{AT} <: SympNet{AT}
    dim::Int
    upscaling_dimension::Int
    n_layers::Int
    act::AT
    init_upper::Bool

    function GSympNet(dim; upscaling_dimension=2*dim, n_layers=2, activation=tanh, init_upper=true) 
        new{typeof(activation)}(dim, upscaling_dimension, n_layers, activation, init_upper)
    end

        
    function GSympNet(dl::DataLoader; upscaling_dimension=2*dl.input_dim, n_layers=2, activation=tanh, init_upper=true) 
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
    Chain(layers...)
end