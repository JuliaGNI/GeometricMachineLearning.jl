"""
The architecture for a "transformer encoder" is essentially taken from arXiv:2010.11929, but with the difference that 𝐧𝐨 layer normalization is employed.
    This is because we still need to find a generalization of layer normalization to manifolds. 
"""
function Transformer(dim::Integer, n_heads::Integer, L::Integer; 
    activation=tanh, Stiefel::Bool=false, retraction=Geodesic(), add_connection=true, use_bias=true)

    layers = ()
    for _ in 1:L
        layers = (layers..., MultiHeadAttention(dim, n_heads, Stiefel=Stiefel, retraction=retraction, add_connection=add_connection), 
        ResNet(dim, activation, use_bias=use_bias) )
    end

    Chain(layers...)
end