@doc raw"""
The volume-preserving transformer with the Cayley activation function and built-in upscaling. The arguments for the constructor are: 
- `sys_dim::Int`
- `seq_length::Int`: The sequence length of the data fed into the transformer.
- `n_blocks::Int=1` (keyword argument): The number of blocks in one transformer unit (containing linear layers and nonlinear layers). Default is `1`.
- `n_linear::Int=1` (keyword argument): The number of linear `VolumePreservingLowerLayer`s and `VolumePreservingUpperLayer`s in one block. Default is `1`.
- `L::Int=1` (keyword argument): The number of transformer units (default is 2). 
- `activation=tanh` (keyward argument): The activation function (`tanh` by default).
- `init_upper::Bool=false` (keyword argument): Specifies if the network first acts on the ``q`` component. 
- `skew_sym::Bool=false` (keyword argument): specifies if we the weight matrix is skew symmetric or arbitrary.
"""
struct VolumePreservingTransformer{AT} <: TransformerIntegrator 
    sys_dim::Int 
    seq_length::Int
    n_blocks::Int
    n_linear::Int
    L::Int 
    activation::AT
    init_upper::Bool
    skew_sym::Bool
end

function VolumePreservingTransformer(sys_dim::Int, seq_length::Int; n_blocks::Int=1, n_linear::Int=1, activation=tanh, L::Int=2, init_upper::Bool=false, skew_sym::Bool=false)
    VolumePreservingTransformer{typeof(activation)}(sys_dim, seq_length, n_blocks, n_linear, L, activation, init_upper, skew_sym)
end

function Chain(arch::VolumePreservingTransformer)
    layers = ()
    
    for _ in 1:arch.L 
        layers = (layers..., VolumePreservingAttention(arch.sys_dim, arch.seq_length; skew_sym = arch.skew_sym))
        layers = (layers..., Chain(VolumePreservingFeedForward(arch.sys_dim, arch.n_blocks, arch.n_linear, arch.activation; init_upper = arch.init_upper)).layers...)
    end

    Chain(layers...)
end