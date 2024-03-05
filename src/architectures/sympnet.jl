@doc raw"""
SympNet type encompasses GSympNets and LASympnets.

TODO: 
-[ ] add bias to `LASympNet`!
"""
abstract type SympNet{AT} <: NeuralNetworkIntegrator end

@doc raw"""
`LASympNet` is called with **a single input argument**, the **system dimension**, or with an instance of `DataLoader`. Optional input arguments are: 
- `depth::Int`: The number of linear layers that are applied. The default is 5.
- `nhidden::Int`: The number of hidden layers (i.e. layers that are **not** input or output layers). The default is 2.
- `activation`: The activation function that is applied. By default this is `tanh`.
- `init_upper_linear::Bool`: Initialize the linear layer so that it first modifies the $q$-component. The default is `true`.
- `init_upper_act::Bool`: Initialize the activation layer so that it first modifies the $q$-component. The default is `true`.
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
`GSympNet` is called with **a single input argument**, the **system dimension**, or with an instance of `DataLoader`. Optional input arguments are: 
- `upscaling_dimension::Int`: The *upscaling dimension* of the gradient layer. See the documentation for `GradientLayerQ` and `GradientLayerP` for further explanation. The default is `2*dim`.
- `nhidden::Int`: The number of hidden layers (i.e. layers that are **not** input or output layers). The default is 2.
- `activation`: The activation function that is applied. By default this is `tanh`.
- `init_upper::Bool`: Initialize the gradient layer so that it first modifies the $q$-component. The default is `true`.
"""
struct GSympNet{AT, InitUpper} <: SympNet{AT} where {InitUpper} 
    dim::Int
    upscaling_dimension::Int
    nhidden::Int
    act::AT

    function GSympNet(dim; upscaling_dimension=2*dim, nhidden=2, activation=tanh, init_upper=true) 
        new{typeof(activation), init_upper}(dim, upscaling_dimension, nhidden, activation)
    end

        
    function GSympNet(dl::DataLoader; upscaling_dimension=2*dl.input_dim, nhidden=2, activation=tanh, init_upper=true) 
        new{typeof(activation), init_upper}(dl.input_dim, upscaling_dimension, nhidden, activation)
    end
end

@doc raw"""
`Chain` can also be called with a neural network as input.
"""
function Chain(arch::GSympNet{AT, true}) where {AT}
    layers = ()
    for _ in 1:(arch.nhidden+1)
        layers = (layers..., GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.act), GradientLayerP(arch.dim, arch.upscaling_dimension, arch.act))
    end
    Chain(layers...)
end

function Chain(arch::GSympNet{AT, false}) where {AT}
    layers = ()
    for _ in 1:(arch.nhidden+1)
        layers = (layers..., GradientLayerP(arch.dim, arch.upscaling_dimension, arch.act), GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.act))
    end
    Chain(layers...)
end

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `true` and `init_upper_act` is `false`.
"""
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

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `false` and `init_upper_act` is `true`.
"""
function Chain(arch::LASympNet{AT, false, true}) where {AT}
    layers = ()
    for i in 1:arch.nhidden
        for j in 1:arch.depth
            layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
        end
        layers = (layers..., ActivationLayerQ(arch.dim, arch.activation))
        layers = (layers..., ActivationLayerP(arch.dim, arch.activation))
    end
    for j in 1:(arch.depth)
        layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
    end
    Chain(layers...)
end

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `false` and `init_upper_act` is `false`.
"""
function Chain(arch::LASympNet{AT, false, false}) where {AT}
    layers = ()
    for i in 1:arch.nhidden
        for j in 1:arch.depth
            layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
        end
        layers = (layers..., ActivationLayerP(arch.dim, arch.activation))
        layers = (layers..., ActivationLayerQ(arch.dim, arch.activation))
    end
    for j in 1:(arch.depth)
        layers = isodd(j) ? (layers..., LinearLayerP(arch.dim)) : (layers..., LinearLayerQ(arch.dim))
    end
    Chain(layers...)
end

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `true` and `init_upper_act` is `true`.
"""
function Chain(arch::LASympNet{AT, true, true}) where {AT}
    layers = ()
    for i in 1:arch.nhidden
        for j in 1:arch.depth
            layers = isodd(j) ? (layers..., LinearLayerQ(arch.dim)) : (layers..., LinearLayerP(arch.dim))
        end
        layers = (layers..., ActivationLayerQ(arch.dim, arch.activation))
        layers = (layers..., ActivationLayerP(arch.dim, arch.activation))
    end
    for j in 1:(arch.depth)
        layers = isodd(j) ? (layers..., LinearLayerQ(arch.dim)) : (layers..., LinearLayerP(arch.dim))
    end
    Chain(layers...)
end