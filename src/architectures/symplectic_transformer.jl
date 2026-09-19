const st_n_sympnet_default = 2
const st_L_default = 1
const st_sympnet_activation_default = tanh
const st_attention_activation_default::AbstractSoftmax = MatrixSoftmax()
const st_init_upper_default = true
const st_symmetric_default = true

@doc raw"""
    SymplecticTransformer <: TransformerIntegrator

# Constructors

```julia
SymplecticTransformer(sys_dim)
```

Make an instance of `SymplecticTransformer` for a specific system dimension and sequence length. Also see [`LinearSymplecticTransformer`](@ref).

# Arguments 

You can provide the additional optional keyword arguments:
- `n_sympnet::Int = """ * "$(st_n_sympnet_default)`" *
     raw""": The number of sympnet layers in the transformer.
- `upscaling_dimension::Int = 2*dim`: The upscaling that is done by the gradient layer.
- `L::Int = """ * "$(st_L_default)`" * raw""": The number of transformer units.
- `sympnet_activation = """ * "$(st_sympnet_activation_default)`" *
     raw""": The activation function for the SympNet layers.
- `attention_activation = """ * "$(st_attention_activation_default)`" *
     raw""": The activation function for the Attention layers.
- `init_upper::Bool = """ * "$(st_init_upper_default)`" *
     raw""": Specifies if the first layer is a ``q``-type layer (`init_upper=true`) or if it is a ``p``-type layer (`init_upper=false`).
- `symmetric::Bool = """ * "$(st_symmetric_default)`" *
     raw""": Specifies whether the weighting matrix ``A`` of the [`SymplecticAttention`](@ref) layers is symmetric (`symmetric=true`) or arbitrary (`symmetric=false`).

The number of SympNet layers in the network is `2n_sympnet`, i.e. for `n_sympnet = 1` we have one [`GradientLayerQ`](@ref) and one [`GradientLayerP`](@ref).
"""
struct SymplecticTransformer{SAT, AAT, Upscaling} <: TransformerIntegrator
    dim::Int
    transformer_dim::Int
    n_sympnet::Int
    upscaling_dimension::Int
    L::Int
    sympnet_activation::SAT
    attention_activation::AAT
    init_upper::Bool
    symmetric::Bool

    function SymplecticTransformer(dim::Int; transformer_dim::Int = dim,
            n_sympnet::Int = st_n_sympnet_default,
            upscaling_dimension::Int = 2 * dim,
            L::Int = st_L_default,
            sympnet_activation = st_sympnet_activation_default,
            attention_activation = st_attention_activation_default,
            init_upper::Bool = st_init_upper_default,
            symmetric::Bool = st_symmetric_default)
        upscale = transformer_dim ≡ dim ? :NoUpscale : :Upscale
        new{typeof(sympnet_activation), typeof(attention_activation), upscale}(dim,
            transformer_dim,
            n_sympnet,
            upscaling_dimension,
            L,
            sympnet_activation,
            attention_activation,
            init_upper,
            symmetric)
    end
end

function _make_block_for_initialization(layers::Tuple, arch::SymplecticTransformer{AT},
        is_upper_criterion::Function) where {AT}
    for i in 1:arch.n_sympnet
        layers = (layers...,
            if is_upper_criterion(i)
                GradientLayerQ(arch.transformer_dim, arch.upscaling_dimension, arch.sympnet_activation)
            else
                GradientLayerP(arch.transformer_dim, arch.upscaling_dimension, arch.sympnet_activation)
            end
        )
    end

    layers
end

function create_layers_for_transformer_dimension_equal_system_dimension(arch::SymplecticTransformer)
    # if `isodd(i)` and `arch.init_upper==true`, then do `GradientLayerQ`.
    is_upper_criterion = arch.init_upper ? isodd : iseven
    layers = ()
    for _ in 1:arch.L
        layers = (layers...,
            SymplecticAttentionQ(arch.transformer_dim; symmetric = arch.symmetric,
                activation = arch.attention_activation))
        layers = _make_block_for_initialization(layers, arch, is_upper_criterion)
        layers = (layers...,
            SymplecticAttentionP(arch.transformer_dim; symmetric = arch.symmetric,
                activation = arch.attention_activation))
        layers = _make_block_for_initialization(layers, arch, is_upper_criterion)
    end

    layers
end

function create_layers_for_transformer_dimension_unequal_system_dimension(arch::SymplecticTransformer)
    (PSDLayer(arch.dim, arch.transformer_dim),
        create_layers_for_transformer_dimension_equal_system_dimension(arch)...,
        PSDLayer(arch.transformer_dim, arch.dim)
    )
end

function Chain(arch::SymplecticTransformer{SAT, AAT, :NoUpscale}) where {SAT, AAT}
    Chain(create_layers_for_transformer_dimension_equal_system_dimension(arch)...)
end

function Chain(arch::SymplecticTransformer{SAT, AAT, :Upscale}) where {SAT, AAT}
    Chain(create_layers_for_transformer_dimension_unequal_system_dimension(arch)...)
end
