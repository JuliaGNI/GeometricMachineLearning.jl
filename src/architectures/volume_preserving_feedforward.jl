description(::Val{:VPFconstructor}) = raw"""
The constructor is called with the following arguments: 
- `sys_dim::Int`: The system dimension. 
- `n_blocks::Int`: The number of blocks in the neural network (containing linear layers and nonlinear layers). Default is `1`.
- `n_linear::Int`: The number of linear `VolumePreservingLowerLayer`s and `VolumePreservingUpperLayer`s in one block. Default is `1`.
- `activation`: The activation function for the nonlinear layers in a block. 
- `init_upper::Bool=false` (keyword argument): Specifies if the first layer is lower or upper. 
"""

"""
Realizes a volume-preserving neural network as a combination of `VolumePreservingLowerLayer` and `VolumePreservingUpperLayer`. 

## Constructor 

$(description(Val(:VPFconstructor)))
"""
struct VolumePreservingFeedForward{AT, InitLowerUpper} <: NeuralNetworkIntegrator 
    sys_dim::Int 
    n_linear::Int 
    n_blocks::Int 
    activation::AT
end

function VolumePreservingFeedForward(sys_dim::Int, n_blocks::Int=1, n_linear::Int=1, activation=tanh; init_upper::Bool=false)
    if init_upper
        return VolumePreservingFeedForward{typeof(activation), :init_upper}(sys_dim, n_linear, n_blocks, activation)
    else 
        return VolumePreservingFeedForward{typeof(activation), :init_lower}(sys_dim, n_linear, n_blocks, activation)
    end
end

function Chain(arch::VolumePreservingFeedForward{AT, :init_lower}) where AT
    layers = ()
    for _ in 1:arch.n_blocks 
        for __ in 1:(arch.n_linear-1)
            layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; include_bias=false)) 
            layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; include_bias=false))
        end

        # linear layers where the last one includes a bias
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; include_bias=false)) 
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; include_bias=true))

        # nonlinear layers
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, arch.activation; include_bias=true))
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, arch.activation; include_bias=true))
    end

    # linear layers for the output
    layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; include_bias=false))
    layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; include_bias=true))

    Chain(layers...)
end

function Chain(arch::VolumePreservingFeedForward{AT, :init_upper}) where AT
    layers = ()
    for _ in 1:arch.n_blocks 
        for __ in 1:(arch.n_linear-1)
            layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; include_bias=false)) 
            layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; include_bias=false))
        end

        # linear layers where the last one includes a bias
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; include_bias=false)) 
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; include_bias=true))

        # nonlinear layers
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, arch.activation; include_bias=true))
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, arch.activation; include_bias=true))
    end

    # linear layers for the output
    layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; include_bias=false))
    layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; include_bias=true))

    Chain(layers...)
end