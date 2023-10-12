@doc raw"""
The architecture for a "transformer encoder" is essentially taken from arXiv:2010.11929, but with the difference that **no** layer normalization is employed.
This is because we still need to find a generalization of layer normalization to manifolds. 

The transformer is called with the following inputs: 
- `dim`: the dimension of the transformer 
- `n_heads`: the number of heads 
- `L`: the number of **transformer blocks**

In addition we have the following optional arguments: 
- `activation`: the activation function used for the `ResNet` (`tanh` by default)
- `Stiefel::Bool`: if the matrices $P^V$, $P^Q$ and $P^K$ should live on a manifold (`false` by default)
- `retraction`: which retraction should be used (`Geodesic()` by default)
- `add_connection::Bool`: if the input should by added to the ouput after the `MultiHeadAttention` layer is used (`true` by default)
- `use_bias::Bool`: If the `ResNet` should use a bias (`true` by default)
"""
function Transformer(dim::Integer, n_heads::Integer, L::Integer; 
    activation=tanh, Stiefel::Bool=false, retraction=Geodesic(), add_connection=true, use_bias=true)

    layers = ()
    for _ in 1:L
        layers = (layers..., MultiHeadAttention(dim, n_heads, Stiefel=Stiefel, retraction=retraction, add_connection=add_connection), 
        ResNet(dim, activation; use_bias=use_bias) )
    end

    Chain(layers...)
end