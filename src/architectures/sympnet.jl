@doc raw"""
SympNet type encompasses GSympNets and LASympnets.
"""
abstract type SympNet{AT} <: Architecture end

@doc raw"""
`LASympNet` is called with **a single input argument**, the **system dimension**. Optional input arguments are: 
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

    function LASympNet(dim; depth=5, nhidden=2, activation=tanh, init_upper_linear=true, init_upper_act=true) 
        new{typeof(activation), init_upper_linear, init_upper_act}(dim, min(depth,5), nhidden, activation)
    end
end

@inline AbstractNeuralNetworks.dim(arch::LASympNet) = arch.dim

@doc raw"""
`GSympNet` is called with **a single input argument**, the **system dimension**. Optional input arguments are: 
- `upscaling_dimension::Int`: The *upscaling dimension* of the gradient layer. See the documentation for `GradientQ` and `GradientP` for further explanation. The default is `2*dim`.
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
end


@inline dim(arch::GSympNet) = arch.dim

@doc raw"""
`Chain` can also be called with a neural network as input.
"""
function Chain(nn::GSympNet{AT, true}) where {AT}
    layers = ()
    for i in 1:(nn.nhidden+1)
        layers = (layers..., GradientQ(nn.dim, nn.upscaling_dimension, nn.act), GradientP(nn.dim, nn.upscaling_dimension, nn.act))
    end
    Chain(layers...)
end

function Chain(nn::GSympNet{AT, false}) where {AT}
    layers = ()
    for i in 1:(nn.nhidden+1)
        layers = (layers..., GradientP(nn.dim, nn.upscaling_dimension, nn.act), GradientQ(nn.dim, nn.upscaling_dimension, nn.act))
    end
    Chain(layers...)
end

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `true` and `init_upper_act` is `false`.
"""
function Chain(nn::LASympNet{AT, true, false}) where {AT}
    layers = ()
    for i in 1:(nn.nhidden+1)
        for j in 1:(nn.depth)
            layers = isodd(j) ? (layers..., LinearQ(nn.dim)) : (layers..., LinearP(nn.dim))
        end
        layers = (layers..., ActivationP(nn.dim, nn.activation))
        layers = (layers..., ActivationQ(nn.dim, nn.activation))
    end
    Chain(layers...)
end

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `false` and `init_upper_act` is `true`.
"""
function Chain(nn::LASympNet{AT, false, true}) where {AT}
    layers = ()
    for i in 1:(nn.nhidden+1)
        for j in 1:(nn.depth)
            layers = isodd(j) ? (layers..., LinearP(nn.dim)) : (layers..., LinearQ(nn.dim))
        end
        layers = (layers..., ActivationQ(nn.dim, nn.activation))
        layers = (layers..., ActivationP(nn.dim, nn.activation))
    end
    Chain(layers...)
end

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `false` and `init_upper_act` is `false`.
"""
function Chain(nn::LASympNet{AT, false, false}) where {AT}
    layers = ()
    for i in 1:(nn.nhidden+1)
        for j in 1:(nn.depth)
            layers = isodd(j) ? (layers..., LinearP(nn.dim)) : (layers..., LinearQ(nn.dim))
        end
        layers = (layers..., ActivationP(nn.dim, nn.activation))
        layers = (layers..., ActivationQ(nn.dim, nn.activation))
    end
    Chain(layers...)
end

@doc raw"""
Build a chain for an LASympnet for which `init_upper_linear` is `true` and `init_upper_act` is `true`.
"""
function Chain(nn::LASympNet{AT, true, true}) where {AT}
    layers = ()
    for i in 1:(nn.nhidden+1)
        for j in 1:(nn.depth)
            layers = isodd(j) ? (layers..., LinearQ(nn.dim)) : (layers..., LinearP(nn.dim))
        end
        layers = (layers..., ActivationQ(nn.dim, nn.activation))
        layers = (layers..., ActivationP(nn.dim, nn.activation))
    end
    Chain(layers...)
end

#=
function Iterate_Sympnet(nn::NeuralNetwork{<:SympNet}, q0, p0; n_points = DEFAULT_SIZE_RESULTS)

    n_dim = length(q0)
    
    # Array to store the predictions
    q_learned = zeros(n_points,n_dim)
    p_learned = zeros(n_points,n_dim)
    
    # Initialisation
    q_learned[1,:] = q0
    p_learned[1,:] = p0
    
    #Computation of phase space
    for i in 2:n_points
        qp_learned =  nn([q_learned[i-1,:]..., p_learned[i-1,:]...])
        q_learned[i,:] = qp_learned[1:n_dim]
        p_learned[i,:] = qp_learned[(1+n_dim):end]
    end

    return q_learned, p_learned
end
=#