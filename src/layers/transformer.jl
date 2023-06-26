"""
The architecture for a "transformer encoder" is essentially taken from arXiv:2010.11929, but with the difference that 𝐧𝐨 layer normalization is employed.
    This is because we still need to find a generalization of layer normalization to manifolds. 
"""
default_retr = Geodesic()
function Transformer(dim::Integer, n_heads::Integer, L::Integer; 
    activation=tanh, Stiefel::Bool=false, Retraction=default_retr, init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32, use_bias=true)

    model = Lux.Chain(Tuple(map(_ -> (MultiHeadAttention(dim, n_heads, Stiefel=Stiefel, Retraction=Retraction, init_weight=init_weight), 
    ResNet(dim, activation, init_weight=init_weight, init_bias=init_bias, use_bias=use_bias)), 1:L))...)

    model
end