@doc raw"""
This is a transformer neural network for classification purposes. At the moment this is only used for training on MNIST, but can in theory be used for any classification problem.

It has to be called with a `DataLoader` that stores an input and an output tensor. The optional arguments are: 
- `n_heads`: The number of heads in the `MultiHeadAttention` (mha) layers. Default: `7`.
- `n_layers`: The number of transformer layers. Default: `16`.
- `activation`: The activation function. Default: `softmax`.
- `Stiefel`: Wheter the matrices in the mha layers are on the **Stiefel manifold**. 
- `add_connection`: Whether the input is appended to the output of the mha layer. (skip connection)
"""
struct ClassificationTransformer{AT} <: Architecture 
    transformer_dim::Int
    classification_dim::Int
    n_heads::Int 
    n_layers::Int
    activation::AT
    Stiefel::Bool
    add_connection::Bool
    function ClassificationTransformer(transformer_dim::Int, classification_dim::Int, n_heads::Int, n_layers::Int, σ::AT, Stiefel::Bool, add_connection::Bool) where AT
        new{AT}(transformer_dim, classification_dim, n_heads, n_layers, σ, Stiefel, add_connection)
    end
    function ClassificationTransformer(dl::DataLoader{T, BT, CT}; n_heads::Int=7, n_layers::Int=16, activation::AT=softmax, Stiefel::Bool=true, add_connection::Bool=true) where {T, T1, BT<:AbstractArray{T, 3}, CT<:AbstractArray{T1, 3}, AT}
        new{AT}(dl.input_dim, dl.output_dim, n_heads, n_layers, activation, Stiefel, add_connection)
    end
end

function Chain(arch::ClassificationTransformer)
    Chain(
        Transformer(arch.transformer_dim, arch.n_heads, arch.n_layers, Stiefel=arch.Stiefel, add_connection=arch.add_connection).layers...,
        Classification(arch.transformer_dim, arch.classification_dim, arch.activation)
        )
end