@doc raw"""
The regular transformer used as an integrator (multi-step method). 

The constructor is called with the following arguments: 
- `sys_dim::Int`
- `transformer_dim::Int`: the default is `transformer_dim = sys_dim`.
- `n_blocks::Int`: The default is `1`.
- `n_heads::Int`: the number of heads in the multihead attentio layer (default is `n_heads = sys_dim`)
- `L::Int` the number of transformer blocks (default is `L = 2`).
- `upscaling_activation`: by default identity
- `resnet_activation`: by default tanh
- `add_connection:Bool=true` (keyword argument): if the input should be added to the output.
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

function StandardTransformerIntegrator(sys_dim::Int, transformer_dim::Int = sys_dim, n_heads::Int = sys_dim; n_blocks = 1, L::Int = 2, upscaling_activation = identity, resnet_activation = tanh, add_connection::Bool = true)
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