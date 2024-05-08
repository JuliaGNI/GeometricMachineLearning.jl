@doc raw"""
Realizes the linear Symplectic Transformer.

### Constructor: 

The constructor is called with the following arguments
1. `dim::Int`: System dimension 
2. `seq_length::Int`: Number of time steps that the transformer considers. 

Optional keyword arguments:
- `n_sympnet::Int=2`: The number of sympnet layers in the transformer.
- `upscaling_dimension::Int=2*dim`: The upscaling that is done by the gradient layer. 
- `L::Int=1`: The number of transformer units. 
- `activation=tanh`: The activation function for the SympNet layers. 
- `init_upper::Bool=true`: Specifies if the first layer is a ``Q``-type layer (`init_upper=true`) or if it is a ``P``-type layer (`init_upper=false`).
"""
struct LinearSymplecticTransformer{AT} <: TransformerIntegrator where AT 
    dim::Int
    sew_length::Int
    n_sympnet::Int
    upscaling_dimension::Int
    L::Int
    activation::AT
    init_upper::Bool

    function LinearSymplecticTransformer(dim::Int, seq_length::Int; n_sympnet::Int=5, upscaling_dimension::Int=2*dim, L::Int=1, activation=tanh, init_upper::Bool=true)
        new{typeof(tanh)}(dim, seq_length, n_sympnet, upscaling_dimension, L, activation, init_upper)
    end
end

function Chain(arch::LinearSymplecticTransformer{AT, true}) where AT
    layers = ()
    for _ in 1:(arch.nhidden + 1)
        layers = (layers..., LinearSymplecticAttentionQ(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.activation))
        end
        layers = (layers..., LinearSymplecticAttentionP(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerP(arch.dim, arch.upscaling_dimension, arch.activation))
        end
    end
    Chain(layers...)
end

function Chain(arch::LinearSymplecticTransformer{AT, false}) where AT
    layers = ()
    for _ in 1:(arch.nhidden+1)
        layers = (layers..., LinearSymplecticAttentionP(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerP(arch.dim, arch.upscaling_dimension, arch.activation))
        end
        layers = (layers..., LinearSymplecticAttentionQ(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.activation))
        end
    end
    Chain(layers...)
end