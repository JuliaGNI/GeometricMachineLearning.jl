const t_activation_default = tanh
const t_Stiefel_default = false
const t_add_connection_default = false
const t_use_bias_default = true

@doc raw"""
    Transformer(dim, n_heads, L)

Make an instance of the Transformer with `n_heads` for dimension `dim` and `L` blocks.

# Arguments

`Transformer` takes the following optional keyword arguments:
- `activation = """ * "$(t_activation_default)`" * raw""": the activation function used for the [`ResNetLayer`](@ref).
- `Stiefel::Bool = """ * "$(t_Stiefel_default)`" * raw""": if the matrices ``P^V``, ``P^Q`` and ``P^K`` should live on a manifold.
- `add_connection::Bool = """ * "$(t_add_connection_default)`" * raw""": if the input should by added to the ouput after the [`MultiHeadAttention`](@ref) layer.
- `use_bias::Bool = """ * "$(t_use_bias_default)`" * raw""": Specifies if the [`ResNetLayer`](@ref) should use a bias.
"""
function Transformer(dim::Integer, n_heads::Integer, L::Integer; 
                                                                activation = t_activation_default, 
                                                                Stiefel::Bool = t_Stiefel_default, 
                                                                add_connection::Bool = t_add_connection_default, 
                                                                use_bias::Bool = t_use_bias_default)
    layers = ()
    for _ in 1:L
        layers = (layers..., MultiHeadAttention(dim, n_heads, Stiefel=Stiefel, add_connection=add_connection), 
        ResNetLayer(dim, activation; use_bias=use_bias) )
    end

    Chain(layers...)
end