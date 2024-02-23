struct VolumePreservingTransformer{AT, InitUpper, Upscaling} <: TransformerIntegrator 
    sys_dim::Int 
    transformer_dim::Int 
    seq_length::Int
    depth::Int
    L::Int 
    activation::AT
end

@doc raw"""
The volume-preserving transformer with the Cayley activation function and built-in upscaling. The arguments for the constructor are: 
- `sys_dim::Int`
- `seq_length::Int`: The sequence length of the data fed into the transformer.
- `depth::Int`: The number of volume-preserving feedforward layers in one transformer block.
- `transformer_dim::Int`: by default equal to `sys_dim`
- `L::Int`: The number of transformer blocks (default is 2). 
- `activation`: The activation function (`tanh` by default).
- `init_upper::Bool`: Specifies if the network first acts on the ``q`` component. 
"""
function VolumePreservingTransformer(sys_dim::Int, seq_length::Int, depth::Int = 2, transformer_dim::Int = sys_dim, L::Int = 2, activation = tanh, init_upper::Bool=true)
    upscaling = sys_dim == transformer_dim ? false : true
    VolumePreservingTransformer{typeof(tanh), init_upper, upscaling}(sys_dim, transformer_dim, seq_length, depth, L, activation)
end

function Chain(arch::VolumePreservingTransformer{AT, true, true}) where {AT}
    layers = (StiefelLayer(arch.sys_dim, arch.transformer_dim), )
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.transformer_dim, arch.seq_length))
        for __ in 1:arch.depth
            layers = (layers..., VolumePreservingLowerLayer(arch.transformer_dim, arch.activation))
            layers = (layers..., VolumePreservingUpperLayer(arch.transformer_dim, arch.activation))
        end
    end
    layers = (layers..., StiefelLayer(arch.transformer_dim, arch.sys_dim))

    Chain(layers...)
end

function Chain(arch::VolumePreservingTransformer{AT, false, true}) where {AT}
    layers = (StiefelLayer(arch.sys_dim, arch.transformer_dim), )
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.transformer_dim, arch.seq_length))
        for __ in 1:arch.depth
            layers = (layers..., VolumePreservingUpperLayer(arch.transformer_dim, arch.activation))
            layers = (layers..., VolumePreservingLowerLayer(arch.transformer_dim, arch.activation))
        end
    end
    layers = (layers..., StiefelLayer(arch.transformer_dim, arch.sys_dim))
    Chain(layers...)
end

function Chain(arch::VolumePreservingTransformer{AT, true, false}) where {AT}
    layers = ()
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.transformer_dim, arch.seq_length))
        for __ in 1:arch.depth
            layers = (layers..., VolumePreservingLowerLayer(arch.transformer_dim, arch.activation))
            layers = (layers..., VolumePreservingUpperLayer(arch.transformer_dim, arch.activation))
        end
    end

    Chain(layers...)
end

function Chain(arch::VolumePreservingTransformer{AT, false, false}) where {AT}
    layers = ()
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.transformer_dim, arch.seq_length))
        for __ in 1:arch.depth
            layers = (layers..., VolumePreservingUpperLayer(arch.transformer_dim, arch.activation))
            layers = (layers..., VolumePreservingLowerLayer(arch.transformer_dim, arch.activation))
        end
    end
    Chain(layers...)
end