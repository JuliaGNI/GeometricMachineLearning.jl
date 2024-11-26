const vpt_n_blocks_default = 1
const vpt_n_linear_default = 1
const vpt_L_default = 1
const vpt_activation_default = tanh
const vpt_init_upper_default = false
const vpt_skew_sym_default = false

@doc raw"""
    VolumePreservingTransformer(sys_dim, seq_length)

Make an instance of the volume-preserving transformer for a given system dimension and sequence length.

# Arguments

The following are keyword argumetns:
- `n_blocks::Int = """ * "$(vpt_n_blocks_default)`" * raw""": The number of blocks in one transformer unit (containing linear layers and nonlinear layers).
- `n_linear::Int = """ * "$(vpt_n_linear_default)`" * raw""": The number of linear `VolumePreservingLowerLayer`s and `VolumePreservingUpperLayer`s in one block.
- `L::Int = """ * "$(vpt_L_default)`" * raw""": The number of transformer units. 
- `activation = """ * "$(vpt_activation_default)`" * raw""": The activation function.
- `init_upper::Bool = """ * "$(vpt_init_upper_default)`" * raw""": Specifies if the network first acts on the ``q`` component. 
- `skew_sym::Bool = """ * "$(vpt_skew_sym_default)`" * raw""": specifies if we the weight matrix is skew symmetric or arbitrary.
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

function VolumePreservingTransformer(sys_dim::Int, seq_length::Int; n_blocks::Int = vpt_n_blocks_default, 
                                                                    n_linear::Int = vpt_n_linear_default, 
                                                                    activation = vpt_activation_default, 
                                                                    L::Int = vpt_L_default, 
                                                                    init_upper::Bool = vpt_init_upper_default, 
                                                                    skew_sym::Bool = vpt_skew_sym_default)
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