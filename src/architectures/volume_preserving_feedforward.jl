const vpff_n_blocks_default = 1
const vpff_n_linear_default = 1
const vpff_activation_default = tanh
const vpff_init_upper_default = false

@doc raw"""
    VolumePreservingFeedForward(dim)

Make an instance of a volume-preserving feedforward neural network for a specific system dimension.    

This architecture is a composition of [`VolumePreservingLowerLayer`](@ref) and [`VolumePreservingUpperLayer`](@ref). 

# Arguments 

You can provide the constructor with the following additional arguments:
2. `n_blocks::Int = """ * "$(vpff_n_blocks_default)`" * raw""": The number of blocks in the neural network (containing linear layers and nonlinear layers).
3. `n_linear::Int = """ * "$(vpff_n_linear_default)`" * raw""": The number of linear `VolumePreservingLowerLayer`s and `VolumePreservingUpperLayer`s in one block.
4. `activation = """ * "$(vpff_activation_default)`" * raw""": The activation function for the nonlinear layers in a block.

The following is a keyword argument:
- `init_upper::Bool = """ * "$(vpff_init_upper_default)`" * raw""": Specifies if the first layer is lower or upper. 
"""
struct VolumePreservingFeedForward{AT, InitLowerUpper} <: NeuralNetworkIntegrator 
    sys_dim::Int 
    n_linear::Int 
    n_blocks::Int 
    activation::AT
end

function VolumePreservingFeedForward(   sys_dim::Int,  
                                        n_blocks::Int = vpff_n_blocks_default, 
                                        n_linear::Int = vpff_n_linear_default,             
                                        activation = vpff_activation_default;             
                                        init_upper::Bool = vpff_init_upper_default)
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
            layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; use_bias=false)) 
            layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; use_bias=false))
        end

        # linear layers where the last one includes a bias
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; use_bias=false)) 
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; use_bias=true))

        # nonlinear layers
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, arch.activation; use_bias=true))
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, arch.activation; use_bias=true))
    end

    # linear layers for the output
    layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; use_bias=false))
    layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; use_bias=true))

    Chain(layers...)
end

function Chain(arch::VolumePreservingFeedForward{AT, :init_upper}) where AT
    layers = ()
    for _ in 1:arch.n_blocks 
        for __ in 1:(arch.n_linear-1)
            layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; use_bias=false)) 
            layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; use_bias=false))
        end

        # linear layers where the last one includes a bias
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; use_bias=false)) 
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; use_bias=true))

        # nonlinear layers
        layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, arch.activation; use_bias=true))
        layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, arch.activation; use_bias=true))
    end

    # linear layers for the output
    layers = (layers..., VolumePreservingUpperLayer(arch.sys_dim, identity; use_bias=false))
    layers = (layers..., VolumePreservingLowerLayer(arch.sys_dim, identity; use_bias=true))

    Chain(layers...)
end