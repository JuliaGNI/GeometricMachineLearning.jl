const sti_n_blocks_default = 1
const sti_L_default = 2
const sti_upscaling_activation_default = identity
const sti_resnet_activation_default = tanh
const sti_add_connection_default = true

@doc raw"""
    StandardTransformerIntegrator(sys_dim)

Make an instance of `StandardTransformerIntegrator` for a specific system dimension.

Here the standard transformer used as an integrator (see [`TransformerIntegrator`](@ref)). 

It is a composition of [`MultiHeadAttention`](@ref) layers and [`ResNetLayer`](@ref) layers.

# Arguments

The following are optional keyword arguments:
- `transformer_dim::Int = sys_dim`: this is the dimension *after the upscaling*.
- `n_blocks::Int = """ * "$(sti_n_blocks_default)`" * raw""" : the number of [`ResNetLayer`](@ref) blocks.
- `n_heads::Int = sys_dim`: the number of heads in the multihead attention layer.
- `L::Int = """ * "$(sti_L_default)`" * raw""": the number of transformer blocks.
- `upscaling_activation = """ * "$(sti_upscaling_activation_default)`" * raw""": the activation used in the upscaling layer.
- `resnet_activation = """ * "$(sti_resnet_activation_default)`" * raw""": the activation used for the [`ResNetLayer`](@ref).
- `add_connection:Bool = """ * "$(sti_add_connection_default)`" * raw""": specifies if the input should be added to the output.
"""
struct StandardTransformerIntegrator{AT1, AT2} <: TransformerIntegrator
    sys_dim::Int 
    transformer_dim::Int 
    n_heads::Int
    n_blocks::Int
    L::Int
    upsacling_activation::AT1
    resnet_activation::AT2
    add_connection::Bool
end

# function StandardTransformerIntegrator(sys_dim::Int, transformer_dim::Int = sys_dim, n_heads::Int = sys_dim, n_blocks = 1, L::Int = 2, upscaling_activation = identity, resnet_activation = tanh; add_connection::Bool = true)
#     StandardTransformerIntegrator{typeof(upscaling_activation), typeof(resnet_activation)}(sys_dim, transformer_dim, n_heads, n_blocks, L, upscaling_activation, resnet_activation, add_connection)
# end

function StandardTransformerIntegrator(sys_dim::Int;    transformer_dim::Int = sys_dim, 
                                                        n_heads::Int = sys_dim, 
                                                        n_blocks = sti_n_blocks_default, 
                                                        L::Int = sti_L_default, 
                                                        upscaling_activation = sti_upscaling_activation_default, 
                                                        resnet_activation = sti_resnet_activation_default, 
                                                        add_connection::Bool = sti_add_connection_default)
    StandardTransformerIntegrator(sys_dim, transformer_dim, n_heads, n_blocks, L, upscaling_activation, resnet_activation, add_connection)
end

function Chain(arch::StandardTransformerIntegrator)
    layers = arch.sys_dim == arch.transformer_dim ? () : (Dense(arch.sys_dim, arch.transformer_dim, arch.upsacling_activation), )
    for _ in 1:arch.L 
        layers = (layers..., MultiHeadAttention(arch.transformer_dim, arch.n_heads; add_connection = arch.add_connection))
        layers = (layers..., Chain(ResNet(arch.transformer_dim, arch.n_blocks, arch.resnet_activation)).layers...)
    end
    layers = arch.sys_dim == arch.transformer_dim ? layers : (layers..., Dense(arch.transformer_dim, arch.sys_dim, identity))

    Chain(layers...)
end