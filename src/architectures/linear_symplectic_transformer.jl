@doc raw"""
    LinearSymplecticTransformer(sys_dim, seq_length)

Make an instance of `LinearSymplecticTransformer` for a specific system dimension and sequence length.

# Arguments 

You can provide the additional optional keyword arguments:
- `n_sympnet::Int=2`: The number of sympnet layers in the transformer.
- `upscaling_dimension::Int=2*dim`: The upscaling that is done by the gradient layer. 
- `L::Int=1`: The number of transformer units. 
- `activation=tanh`: The activation function for the SympNet layers. 
- `init_upper::Bool=true`: Specifies if the first layer is a ``Q``-type layer (`init_upper=true`) or if it is a ``P``-type layer (`init_upper=false`).
"""
struct LinearSymplecticTransformer{AT} <: TransformerIntegrator where AT 
    dim::Int
    seq_length::Int
    n_sympnet::Int
    upscaling_dimension::Int
    L::Int
    activation::AT
    init_upper::Bool

    function LinearSymplecticTransformer(dim::Int, seq_length::Int; n_sympnet::Int=2, upscaling_dimension::Int=2*dim, L::Int=1, activation=tanh, init_upper::Bool=true)
        new{typeof(tanh)}(dim, seq_length, n_sympnet, upscaling_dimension, L, activation, init_upper)
    end
end

function _make_block_for_initialization(layers::Tuple, arch::LinearSymplecticTransformer{AT}, is_upper_criterion::Function) where AT
    for i in 1:arch.n_sympnet
        layers = (layers..., 
            if is_upper_criterion(i)
                GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.activation)
            else
                GradientLayerP(arch.dim, arch.upscaling_dimension, arch.activation)
            end
        )
    end

    layers
end


function Chain(arch::LinearSymplecticTransformer{AT}) where AT
    # if `isodd(i)` and `arch.init_upper==true`, then do `GradientLayerQ`.
    is_upper_criterion = arch.init_upper ? isodd : iseven
    layers = ()
    for _ in 1:arch.L
        layers = (layers..., LinearSymplecticAttentionQ(arch.dim, arch.seq_length))
        layers = _make_block_for_initialization(layers, arch, is_upper_criterion)
        layers = (layers..., LinearSymplecticAttentionP(arch.dim, arch.seq_length))
        layers = _make_block_for_initialization(layers, arch, is_upper_criterion)
    end

    Chain(layers...)
end