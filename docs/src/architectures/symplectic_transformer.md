# The Symplectic Transformer

The symplectic transformer is, like the [standard transformer](@ref "Standard Transformer"), a combination of *attention layers* and *feedforward layers*. The difference is that the attention layers are not [multihead attention layers](@ref "Multihead Attention") and the feedforward layers are not standard [`ResNetLayer`](@ref)s, but [symplectic attention layers](@ref "Symplectic Attention") and [sympnet layers](@ref "SympNet Layers").

## Library Functions 

```@docs
SymplecticTransformer
```