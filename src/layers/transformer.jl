@doc raw"""
    Transformer(dim, n_heads, L)

Make an instance of the Transformer with `n_heads` for dimension `dim` and `L` blocks.

The architecture for a "transformer encoder" is essentially taken from arXiv:2010.11929, but with the difference that **no** layer normalization is employed.
This is because we still need to find a generalization of layer normalization to manifolds. 

The transformer is called with the following inputs: 
1. `dim`: the dimension of the transformer 
2. `n_heads`: the number of heads 
3. `L`: the number of **transformer blocks**

# Arguments

`Transformer` takes the following optional keyword arguments:
- `activation=tanh`: the activation function used for the `ResNet`.
- `Stiefel::Bool=false`: if the matrices $P^V$, $P^Q$ and $P^K$ should live on a manifold.
- `add_connection::Bool=true`: if the input should by added to the ouput after the `MultiHeadAttention` layer is used.
- `use_bias::Bool=true`: Specifies if the `ResNet` should use a bias.
"""
function Transformer(dim::Integer, n_heads::Integer, L::Integer; 
    activation=tanh, Stiefel::Bool=false, add_connection=true, use_bias=true)

    layers = ()
    for _ in 1:L
        layers = (layers..., MultiHeadAttention(dim, n_heads, Stiefel=Stiefel, add_connection=add_connection), 
        ResNetLayer(dim, activation; use_bias=use_bias) )
    end

    Chain(layers...)
end