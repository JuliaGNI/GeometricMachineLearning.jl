struct VolumePreservingTransformer{AT, InitUpper} <: TransformerIntegrator 
    sys_dim::Int 
    transformer_dim::Int 
    upscaling_dimension::Int 
    seq_length::Int
    L::Int 
    activation::AT
end

@doc raw"""
The volume-preserving transformer with the Cayley activation function and built-in upscaling. The arguments for the constructor are: 
- `sys_dim::Int`
- `seq_length::Int`: The sequence length of the data fed into the transformer.
- `transformer_dim::Int`: by default equal to `sys_dim`
- `upscaling_dimension::Int`: by default equal to `2 * transformer_dim`
- `L::Int`: The number of transformer blocks (default is 2). 
- `activation`: The activation function (`tanh` by default).
- `init_upper::Bool`: Specifies if the network first acts on the ``q`` component. 
"""
function VolumePreservingTransformer(sys_dim::Int, seq_length::Int, transformer_dim::Int = sys_dim, upscaling_dimension::Int = 2 * transformer_dim, L::Int = 2, activation = tanh, init_upper::Bool=true)
    VolumePreservingTransformer{typeof(tanh), init_upper}(sys_dim, transformer_dim, upscaling_dimension, seq_length, L, activation)
end

function Chain(arch::VolumePreservingTransformer{AT, true}) where {AT}
    layers = (PSDLayer(arch.sys_dim, arch.transformer_dim), )
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.transformer_dim, arch.seq_length))
        layers = (layers..., GradientLayerQ(arch.transformer_dim, 2 * arch.transformer_dim, arch.activation))
        layers = (layers..., GradientLayerP(arch.transformer_dim, 2 * arch.transformer_dim, arch.activation))
    end
    layers = (layers..., PSDLayer(arch.transformer_dim, arch.sys_dim))

    Chain(layers...)
end

function Chain(arch::VolumePreservingTransformer{AT, false}) where {AT}
    layers = (PSDLayer(arch.sys_dim, arch.transformer_dim), )
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.transformer_dim, arch.seq_length))
        layers = (layers..., GradientLayerP(arch.transformer_dim, 2 * arch.transformer_dim, arch.activation))
        layers = (layers..., GradientLayerQ(arch.transformer_dim, 2 * arch.transformer_dim, arch.activation))
    end
    layers = (layers..., PSDLayer(arch.transformer_dim, arch.sys_dim))
    Chain(layers...)
end