@doc raw"""
The regular transformer used as an integrator (multi-step method). 

The constructor is called with the following arguments: 
- `sys_dim::Int`
- `transformer_dim::Int`: the defualt is `transformer_dim = sys_dim`.
- `n_heads::Int`: the number of heads in the multihead attentio layer (default is `n_heads = sys_dim`)
- `L::Int` the number of transformer blocks (default is `L = 2`).
- `upscaling_activation`: by default identity
- `resnet_activation`: by default tanh
"""
struct RegularTransformerIntegrator{AT1, AT2} <: TransformerIntegrator
    sys_dim::Int 
    transformer_dim::Int 
    n_heads::Int
    L::Int
    upsacling_activation::AT1
    resnet_activation::AT2

    function RegularTransformerIntegrator(sys_dim::Int, transformer_dim::Int = sys_dim, n_heads::Int = sys_dim, L::Int = 2, upscaling_activation = identity, resnet_activation = tanh)
        new{typeof(upscaling_activation), typeof(resnet_activation)}(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)
    end
end


function Chain(arch::RegularTransformerIntegrator)
    layers = (Dense(arch.sys_dim, arch.transformer_dim, arch.upsacling_activation), )
    for _ in 1:arch.L 
        layers = (layers..., MultiHeadAttention(arch.transformer_dim, arch.n_heads))
        layers = (layers..., ResNet(arch.transformer_dim, arch.resnet_activation))
    end
    layers = (layers..., Dense(arch.transformer_dim, arch.sys_dim, identity))

    Chain(layers...)
end