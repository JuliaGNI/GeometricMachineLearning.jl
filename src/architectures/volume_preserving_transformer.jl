@doc raw"""
The volume-preserving transformer with the Cayley activation function and built-in upscaling. The arguments for the constructor are: 
- `sys_dim::Int`
- `seq_length::Int`: The sequence length of the data fed into the transformer.
- `n_blocks::Int`: The number of blocks in one transformer unit (containing linear layers and nonlinear layers). Default is `1`.
- `n_linear::Int`: The number of linear `VolumePreservingLowerLayer`s and `VolumePreservingUpperLayer`s in one block. Default is `1`.
- `L::Int`: The number of transformer units (default is 2). 
- `activation`: The activation function (`tanh` by default).
- `init_upper::Bool`: Specifies if the network first acts on the ``q`` component. 
"""
struct VolumePreservingTransformer{AT} <: TransformerIntegrator 
    sys_dim::Int 
    seq_length::Int
    n_blocks::Int
    n_linear::Int
    L::Int 
    activation::AT
    init_upper::Bool

    function VolumePreservingTransformer(sys_dim::Int, seq_length::Int, n_blocks::Int=1, n_linear::Int=1, L::Int=2, activation=tanh, init_upper::Bool=false)
        return new{typeof(tanh)}(sys_dim, seq_length, n_blocks, n_linear, L, activation, init_upper)
    end
end

function Chain(arch::VolumePreservingTransformer)
    layers = ()
    
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.sys_dim, arch.seq_length))
        layers = (layers..., Chain(VolumePreservingFeedForward(arch.sys_dim, arch.n_blocks, arch.n_linear, arch.tanh; init_upper=init_upper)).layers...)
    end

    Chain(layers...)
end