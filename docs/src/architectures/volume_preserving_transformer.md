# Volume-Preserving Transformer

The volume-preserving transformer is, similar to the standard transformer, a combination of two different neural networks: a [volume-preserving attention layer](@ref "Volume-Preserving Attention") and a [volume-preserving feedforward layer](@ref "Volume-Preserving Feedforward Neural Network"). It is visualized below:

```@example 
Main.include_graphics("../tikz/vp_transformer") # hide
```

## Library Functions 

```@docs; canonical=false
VolumePreservingTransformer
```

## References 

```@bibliography
Pages = []
Canonical = false

brantner2024volume
```