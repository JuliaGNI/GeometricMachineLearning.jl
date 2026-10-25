const lst_n_sympnet_default = 2
const lst_L_default = 1
const lst_activation_default = tanh
const lst_init_upper_default = true

@doc raw"""
    LinearSymplecticTransformer(sys_dim, seq_length)

Make an instance of `LinearSymplecticTransformer` for a specific system dimension and sequence length.

# Arguments 

You can provide the additional optional keyword arguments:
- `n_sympnet::Int = """ * "($lst_n_sympnet_default)`" *
     raw""": The number of sympnet layers in the transformer.
- `upscaling_dimension::Int = 2*dim`: The upscaling that is done by the gradient layer. 
- `L::Int = """ * "$(lst_L_default)`" * raw""": The number of transformer units. 
- `activation = """ * "$(lst_activation_default)`" *
     raw""": The activation function for the SympNet layers. 
- `init_upper::Bool=true`: Specifies if the first layer is a ``q``-type layer (`init_upper=true`) or if it is a ``p``-type layer (`init_upper=false`).

Each transformer unit has `n_sympnet` gradient layers after its ``q`` attention layer and `n_sympnet` after its ``p`` attention layer, so one unit has `2n_sympnet` SympNet layers. In each group of `n_sympnet` layers the type alternates between [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref), starting with the type that `init_upper` selects. For `n_sympnet = 1` and `init_upper = true` each group is one [`GradientLayerQ`](@ref).
"""
struct LinearSymplecticTransformer{AT} <: TransformerIntegrator where {AT}
    dim::Int
    seq_length::Int
    n_sympnet::Int
    upscaling_dimension::Int
    L::Int
    activation::AT
    init_upper::Bool

    function LinearSymplecticTransformer(
            dim::Int, seq_length::Int; n_sympnet::Int = lst_n_sympnet_default,
            upscaling_dimension::Int = 2 * dim,
            L::Int = lst_L_default,
            activation = lst_activation_default,
            init_upper::Bool = lst_init_upper_default)
        new{typeof(activation)}(
            dim, seq_length, n_sympnet, upscaling_dimension, L, activation, init_upper)
    end
end

function Chain(arch::LinearSymplecticTransformer{AT}) where {AT}
    gradient_layers = _gradient_layers(
        arch.dim, arch.upscaling_dimension, arch.activation, arch.n_sympnet, arch.init_upper)
    layers = ()
    for _ in 1:arch.L
        layers = (layers..., LinearSymplecticAttentionQ(arch.dim, arch.seq_length),
            gradient_layers..., LinearSymplecticAttentionP(arch.dim, arch.seq_length),
            gradient_layers...)
    end

    Chain(layers...)
end
