"""
The architecture for a "transformer encoder" is essentially taken from arXiv:2010.11929, but with the difference that 𝐧𝐨 layer normalization is employed.
    This is because we still need to find a generalization of layer normalization to manifolds. 
"""
default_retr = Geodesic()
function Transformer(dim::Integer, n_heads::Integer, L::Integer; 
    Stiefel::Bool=false, Retraction=default_retr, init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32, use_bias=true)

    model = Chain(Tuple(map(_ -> (MultiHeadAttention(dim, n_heads, Stiefel=Stiefel, Retraction=Retraction, init_weight=Lux.glorot_uniform), 
    ResNet(dim, dim, init_weight=Lux.glorot_uniform, init_bias=init.bias, use_bias=use_bias)), 1:L))...)
end